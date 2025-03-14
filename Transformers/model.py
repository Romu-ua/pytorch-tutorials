import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):
    """
    マルチヘッド注意機構のサブレイヤー
    """

    def __init__(self, d_model, n_heads, d_queries, d_values, dropout, in_decoder=False):
        """
        :param d_model: Transformerモデル全体で使用されるベクトルの次元数（このサブレイヤーの入力および出力のサイズ）
        :param n_heads: マルチヘッド注意機構のヘッド数
        :param d_queries: クエリベクトルの次元数（キーの次元数としても用いられる）
        :param d_values: バリューベクトルの次元数
        :param dropout: ドロップアウト確率
        :param in_decoder: このマルチヘッド注意機構がデコーダ内にあるかどうか
        """
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_queries = d_queries
        self.d_values = d_values
        self.d_keys = d_queries  # キーベクトルの次元数はクエリと同じ。これにより類似度計算（ドット積）が可能になる

        self.in_decoder = in_decoder

        # 入力のクエリシーケンスから (n_heads 個の) クエリを生成するための線形変換層
        self.cast_queries = nn.Linear(d_model, n_heads * d_queries)

        # 参照シーケンスから (n_heads 個の) キーとバリューを生成するための線形変換層
        self.cast_keys_values = nn.Linear(d_model, n_heads * (d_queries + d_values))

        # 注意機構によって重み付けされたベクトル群を、入力クエリと同じサイズの出力ベクトルに変換する線形変換層
        self.cast_output = nn.Linear(n_heads * d_values, d_model)

        # ソフトマックス層
        self.softmax = nn.Softmax(dim=-1)

        # レイヤ正規化層
        self.layer_norm = nn.LayerNorm(d_model)

        # ドロップアウト層
        self.apply_dropout = nn.Dropout(dropout)


	def forward(self, query_sequences, key_value_sequences, key_value_sequence_lengths):
		"""
		順伝播（フォワードプロパゲーション）を行う関数

		:param query_sequences: 入力のクエリシーケンス（テンソルサイズ: (N, クエリのパッド長, d_model)）
		:param key_value_sequences: クエリ対象となるシーケンス（テンソルサイズ: (N, キー・バリューのパッド長, d_model)）
		:param key_value_sequence_lengths: キー・バリューシーケンスの実際の長さ（パディングを無視するため、テンソルサイズ: (N)）
		:return: クエリシーケンスに対して注意重み付けされた出力シーケンス（テンソルサイズ: (N, クエリのパッド長, d_model)）
		"""
		batch_size = query_sequences.size(0)  # バッチサイズ (N)
		query_sequence_pad_length = query_sequences.size(1)  # クエリシーケンスのパッド長
		key_value_sequence_pad_length = key_value_sequences.size(1)  # キー・バリューシーケンスのパッド長

		# これは自己注意かどうか判定
		self_attention = torch.equal(key_value_sequences, query_sequences)

		# 後で残差接続のために入力を保存
		input_to_add = query_sequences.clone()

		# レイヤ正規化を適用
		query_sequences = self.layer_norm(query_sequences)  # (N, クエリのパッド長, d_model)
		# 自己注意の場合、キー・バリューシーケンスにも同じ正規化を適用する
		# 自己注意でない場合は、エンコーダの最終層で既に正規化されている
		if self_attention:
			key_value_sequences = self.layer_norm(key_value_sequences)  # (N, キー・バリューのパッド長, d_model)

		# 入力シーケンスをクエリ、キー、バリューに射影
		queries = self.cast_queries(query_sequences)  # (N, クエリのパッド長, n_heads * d_queries)
		keys, values = self.cast_keys_values(key_value_sequences).split(split_size=self.n_heads * self.d_keys,
																		dim=-1)
		# keys: (N, キー・バリューのパッド長, n_heads * d_keys)
		# values: (N, キー・バリューのパッド長, n_heads * d_values)

		# 最後の次元を n_heads 個のサブスペースに分割する
		queries = queries.contiguous().view(batch_size, query_sequence_pad_length, self.n_heads, self.d_queries)
		# (N, クエリのパッド長, n_heads, d_queries)
		keys = keys.contiguous().view(batch_size, key_value_sequence_pad_length, self.n_heads, self.d_keys)
		# (N, キー・バリューのパッド長, n_heads, d_keys)
		values = values.contiguous().view(batch_size, key_value_sequence_pad_length, self.n_heads, self.d_values)
		# (N, キー・バリューのパッド長, n_heads, d_values)

		# 軸を入れ替えて、最後の2次元がシーケンス長とクエリ/キー/バリューの次元になるようにする
		# その後、バッチ次元とヘッド次元を統合して3次元テンソルに変換（バッチ行列積のための準備）
		queries = queries.permute(0, 2, 1, 3).contiguous().view(-1, query_sequence_pad_length, self.d_queries)
		# (N * n_heads, クエリのパッド長, d_queries)
		keys = keys.permute(0, 2, 1, 3).contiguous().view(-1, key_value_sequence_pad_length, self.d_keys)
		# (N * n_heads, キー・バリューのパッド長, d_keys)
		values = values.permute(0, 2, 1, 3).contiguous().view(-1, key_value_sequence_pad_length, self.d_values)
		# (N * n_heads, キー・バリューのパッド長, d_values)

		# マルチヘッド注意を実行

		# ドット積計算を実施
		attention_weights = torch.bmm(queries, keys.permute(0, 2, 1))
		# (N * n_heads, クエリのパッド長, キー・バリューのパッド長)

		# ドット積のスケーリング
		attention_weights = (1. / math.sqrt(self.d_keys)) * attention_weights
		# (N * n_heads, クエリのパッド長, キー・バリューのパッド長)

		# ソフトマックスを計算する前に、特定のキーへのアテンションを抑制するための処理

		# マスク 1: パッドとなっているキーの位置をマスクする
		not_pad_in_keys = torch.LongTensor(range(key_value_sequence_pad_length)).unsqueeze(0).unsqueeze(0).expand_as(attention_weights).to(device)
		# (N * n_heads, クエリのパッド長, キー・バリューのパッド長)
		not_pad_in_keys = not_pad_in_keys < key_value_sequence_lengths.repeat_interleave(self.n_heads).unsqueeze(1).unsqueeze(2).expand_as(attention_weights)
		# (N * n_heads, クエリのパッド長, キー・バリューのパッド長)
		# 注意: PyTorchは比較演算においてsingleton次元を自動ブロードキャストする

		# マスク対象の重みを非常に大きな負の値に設定し、ソフトマックス後にゼロとなるようにする
		attention_weights = attention_weights.masked_fill(~not_pad_in_keys, -float('inf'))
		# (N * n_heads, クエリのパッド長, キー・バリューのパッド長)

		# マスク 2: デコーダ内の自己注意の場合、クエリより未来のキー（後続の単語）をマスクする
		if self.in_decoder and self_attention:
			# 位置 [n, i, j] が有効なのは、 j <= i の場合のみ
			# torch.tril() を使い、下三角行列を抽出することで j > i の位置を0に設定する
			not_future_mask = torch.ones_like(attention_weights).tril().bool().to(device)
			# (N * n_heads, クエリのパッド長, キー・バリューのパッド長)

			# マスク対象の重みを非常に大きな負の値に設定する
			attention_weights = attention_weights.masked_fill(~not_future_mask, -float('inf'))
			# (N * n_heads, クエリのパッド長, キー・バリューのパッド長)

		# キー方向に沿ってソフトマックスを計算
		attention_weights = self.softmax(attention_weights)
		# (N * n_heads, クエリのパッド長, キー・バリューのパッド長)

		# ドロップアウトを適用
		attention_weights = self.apply_dropout(attention_weights)
		# (N * n_heads, クエリのパッド長, キー・バリューのパッド長)

		# ソフトマックスの重みを用いてバリューの加重和を計算（注意の出力シーケンスを得る）
		sequences = torch.bmm(attention_weights, values)
		# (N * n_heads, クエリのパッド長, d_values)

		# バッチ次元とヘッド次元を統合して元の軸の順序に戻す
		sequences = sequences.contiguous().view(batch_size, self.n_heads, query_sequence_pad_length, self.d_values).permute(0, 2, 1, 3)
		# (N, クエリのパッド長, n_heads, d_values)

		# 各ヘッドごとの出力（d_values次元）を連結する
		sequences = sequences.contiguous().view(batch_size, query_sequence_pad_length, -1)
		# (N, クエリのパッド長, n_heads * d_values)

		# 連結されたサブスペースのシーケンスを、線形変換して d_model 次元の出力に変換する
		sequences = self.cast_output(sequences)
		# (N, クエリのパッド長, d_model)

		# ドロップアウトと残差接続を適用
		sequences = self.apply_dropout(sequences) + input_to_add
		# (N, クエリのパッド長, d_model)

		return sequences


class PositionWiseFCNetwork(nn.Module):
    """
    位置ごとのフィードフォワードネットワークのサブレイヤー
    """

    def __init__(self, d_model, d_inner, dropout):
        """
        :param d_model: Transformerモデル全体で使用されるベクトルの次元数（このサブレイヤーの入力および出力のサイズ）
        :param d_inner: 中間層の次元数
        :param dropout: ドロップアウト確率
        """
        super(PositionWiseFCNetwork, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner

        # レイヤ正規化層
        self.layer_norm = nn.LayerNorm(d_model)

        # 入力サイズから中間サイズに射影する線形層
        self.fc1 = nn.Linear(d_model, d_inner)

        # ReLU活性化関数
        self.relu = nn.ReLU()

        # 中間サイズから出力サイズ（入力と同じ）に射影する線形層
        self.fc2 = nn.Linear(d_inner, d_model)

        # ドロップアウト層
        self.apply_dropout = nn.Dropout(dropout)

    def forward(self, sequences):
        """
        順伝播（フォワードプロパゲーション）を行う関数

        :param sequences: 入力シーケンス（テンソルサイズ: (N, パッド長, d_model)）
        :return: 変換後の出力シーケンス（テンソルサイズ: (N, パッド長, d_model)）
        """
        # 残差接続用に入力を保存
        input_to_add = sequences.clone()  # (N, パッド長, d_model)

        # レイヤ正規化を適用
        sequences = self.layer_norm(sequences)  # (N, パッド長, d_model)

        # 位置ごとに変換を実施
        sequences = self.apply_dropout(self.relu(self.fc1(sequences)))  # (N, パッド長, d_inner)
        sequences = self.fc2(sequences)  # (N, パッド長, d_model)

        # ドロップアウトと残差接続を適用
        sequences = self.apply_dropout(sequences) + input_to_add  # (N, パッド長, d_model)

        return sequences

class Encoder(nn.Module):
    """
    エンコーダー
    """

    def __init__(self, vocab_size, positional_encoding, d_model, n_heads, d_queries, d_values, d_inner, n_layers,
                 dropout):
        """
        :param vocab_size: （共有される）語彙数
        :param positional_encoding: 最大パッド長までの位置エンコーディング
        :param d_model: Transformerモデル全体で使用されるベクトルの次元数（エンコーダの入力および出力のサイズ）
        :param n_heads: マルチヘッド注意機構のヘッド数
        :param d_queries: マルチヘッド注意機構におけるクエリ（キー）のベクトルの次元数
        :param d_values: マルチヘッド注意機構におけるバリューのベクトルの次元数
        :param d_inner: 位置ごとのフィードフォワード層の中間次元数
        :param n_layers: エンコーダ内の [マルチヘッド注意 + 位置ごとのフィードフォワード] 層の数
        :param dropout: ドロップアウト確率
        """
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout

        # 埋め込み層
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置エンコーディングのテンソルは更新されないように設定（勾配計算が行われない）
        self.positional_encoding.requires_grad = False

        # エンコーダ層を作成（各層はマルチヘッド注意と位置ごとのフィードフォワードから構成）
        self.encoder_layers = nn.ModuleList([self.make_encoder_layer() for i in range(n_layers)])

        # ドロップアウト層
        self.apply_dropout = nn.Dropout(dropout)

        # レイヤ正規化層
        self.layer_norm = nn.LayerNorm(d_model)

    def make_encoder_layer(self):
        """
        エンコーダ内の1層を作成する。
        1層は、マルチヘッド注意サブレイヤーと位置ごとのフィードフォワードサブレイヤーの組み合わせから構成される。
        """
        # サブレイヤーのModuleListを作成
        encoder_layer = nn.ModuleList([
            MultiHeadAttention(d_model=self.d_model,
                               n_heads=self.n_heads,
                               d_queries=self.d_queries,
                               d_values=self.d_values,
                               dropout=self.dropout,
                               in_decoder=False),
            PositionWiseFCNetwork(d_model=self.d_model,
                                  d_inner=self.d_inner,
                                  dropout=self.dropout)
        ])

        return encoder_layer

    def forward(self, encoder_sequences, encoder_sequence_lengths):
        """
        順伝播（フォワードプロパゲーション）を行う関数

        :param encoder_sequences: ソース言語のシーケンス（テンソルサイズ: (N, パッド長)）
        :param encoder_sequence_lengths: これらのシーケンスの実際の長さ（パディングを無視するため、テンソルサイズ: (N)）
        :return: エンコードされたソース言語のシーケンス（テンソルサイズ: (N, パッド長, d_model)）
        """
        pad_length = encoder_sequences.size(1)  # このバッチのパッド長（バッチごとに異なる）

        # 語彙の埋め込みと位置エンコーディングの和を計算
        encoder_sequences = self.embedding(encoder_sequences) * math.sqrt(self.d_model) + \
                            self.positional_encoding[:, :pad_length, :].to(device)
        # 結果のサイズ: (N, パッド長, d_model)

        # ドロップアウトを適用
        encoder_sequences = self.apply_dropout(encoder_sequences)  # (N, パッド長, d_model)

        # エンコーダ層を順次適用
        for encoder_layer in self.encoder_layers:
            # 各層のサブレイヤーを適用
            encoder_sequences = encoder_layer[0](query_sequences=encoder_sequences,
                                                 key_value_sequences=encoder_sequences,
                                                 key_value_sequence_lengths=encoder_sequence_lengths)
            # 結果のサイズ: (N, パッド長, d_model)
            encoder_sequences = encoder_layer[1](sequences=encoder_sequences)
            # 結果のサイズ: (N, パッド長, d_model)

        # 最終的にレイヤ正規化を適用
        encoder_sequences = self.layer_norm(encoder_sequences)  # (N, パッド長, d_model)

        return encoder_sequences


class Decoder(nn.Module):
    """
    デコーダー
    """

    def __init__(self, vocab_size, positional_encoding, d_model, n_heads, d_queries, d_values, d_inner, n_layers,
                 dropout):
        """
        :param vocab_size: （共有される）語彙数
        :param positional_encoding: 最大パッド長までの位置エンコーディング
        :param d_model: Transformerモデル全体で使用されるベクトルの次元数（デコーダの入力および出力のサイズ）
        :param n_heads: マルチヘッド注意機構のヘッド数
        :param d_queries: マルチヘッド注意機構におけるクエリ（キー）のベクトルの次元数
        :param d_values: マルチヘッド注意機構におけるバリューのベクトルの次元数
        :param d_inner: 位置ごとのフィードフォワード層の中間次元数
        :param n_layers: デコーダ内の [自己注意 + エンコーダ-デコーダ注意 + 位置ごとのフィードフォワード] 層の数
        :param dropout: ドロップアウト確率
        """
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout

        # 埋め込み層
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置エンコーディングのテンソルは更新されないように設定（勾配計算は行われない）
        self.positional_encoding.requires_grad = False

        # デコーダ層の作成
        self.decoder_layers = nn.ModuleList([self.make_decoder_layer() for i in range(n_layers)])

        # ドロップアウト層
        self.apply_dropout = nn.Dropout(dropout)

        # レイヤ正規化層
        self.layer_norm = nn.LayerNorm(d_model)

        # 語彙に対するロジット（対数尤度）を計算するための出力用線形層
        self.fc = nn.Linear(d_model, vocab_size)

    def make_decoder_layer(self):
        """
        デコーダ内の1層を作成する。
        1層は、2つのマルチヘッド注意サブレイヤー（自己注意とエンコーダ-デコーダ注意）と位置ごとのフィードフォワードサブレイヤーから構成される。
        """
        # サブレイヤーのModuleListを作成
        decoder_layer = nn.ModuleList([
            MultiHeadAttention(d_model=self.d_model,
                               n_heads=self.n_heads,
                               d_queries=self.d_queries,
                               d_values=self.d_values,
                               dropout=self.dropout,
                               in_decoder=True),
            MultiHeadAttention(d_model=self.d_model,
                               n_heads=self.n_heads,
                               d_queries=self.d_queries,
                               d_values=self.d_values,
                               dropout=self.dropout,
                               in_decoder=True),
            PositionWiseFCNetwork(d_model=self.d_model,
                                  d_inner=self.d_inner,
                                  dropout=self.dropout)
        ])

        return decoder_layer

    def forward(self, decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths):
        """
        順伝播（フォワードプロパゲーション）を行う関数

        :param decoder_sequences: ターゲット言語（デコーダ）のシーケンス（テンソルサイズ: (N, パッド長)）
        :param decoder_sequence_lengths: これらのシーケンスの実際の長さ（パディングを無視するため、テンソルサイズ: (N)）
        :param encoder_sequences: エンコードされたソース言語のシーケンス（テンソルサイズ: (N, エンコーダのパッド長, d_model)）
        :param encoder_sequence_lengths: ソースシーケンスの実際の長さ（テンソルサイズ: (N)）
        :return: デコードされたターゲット言語のシーケンス（テンソルサイズ: (N, パッド長, vocab_size)）
        """
        pad_length = decoder_sequences.size(1)  # このバッチのパッド長（バッチごとに異なる）

        # 語彙の埋め込みと位置エンコーディングの和を計算
        decoder_sequences = self.embedding(decoder_sequences) * math.sqrt(self.d_model) + \
                            self.positional_encoding[:, :pad_length, :].to(device)
        # 結果のサイズ: (N, パッド長, d_model)

        # ドロップアウトを適用
        decoder_sequences = self.apply_dropout(decoder_sequences)

        # デコーダ層の適用
        for decoder_layer in self.decoder_layers:
            # サブレイヤー1: 自己注意
            decoder_sequences = decoder_layer[0](query_sequences=decoder_sequences,
                                                  key_value_sequences=decoder_sequences,
                                                  key_value_sequence_lengths=decoder_sequence_lengths)
            # 結果のサイズ: (N, パッド長, d_model)

            # サブレイヤー2: エンコーダ-デコーダ注意
            decoder_sequences = decoder_layer[1](query_sequences=decoder_sequences,
                                                  key_value_sequences=encoder_sequences,
                                                  key_value_sequence_lengths=encoder_sequence_lengths)
            # 結果のサイズ: (N, パッド長, d_model)

            # サブレイヤー3: 位置ごとのフィードフォワード
            decoder_sequences = decoder_layer[2](sequences=decoder_sequences)
            # 結果のサイズ: (N, パッド長, d_model)

        # 最終的にレイヤ正規化を適用
        decoder_sequences = self.layer_norm(decoder_sequences)  # (N, パッド長, d_model)

        # 出力線形層を用いて、語彙に対するロジット（対数尤度）を計算
        decoder_sequences = self.fc(decoder_sequences)  # (N, パッド長, vocab_size)

        return decoder_sequences


class Transformer(nn.Module):
    """
    Transformerネットワーク
    """

    def __init__(self, vocab_size, positional_encoding, d_model=512, n_heads=8, d_queries=64, d_values=64,
                 d_inner=2048, n_layers=6, dropout=0.1):
        """
        :param vocab_size: （共有される）語彙数
        :param positional_encoding: 最大パッド長までの位置エンコーディング
        :param d_model: Transformerモデル全体で使用されるベクトルの次元数
        :param n_heads: マルチヘッド注意機構のヘッド数
        :param d_queries: マルチヘッド注意機構におけるクエリ（およびキー）のベクトルの次元数
        :param d_values: マルチヘッド注意機構におけるバリューのベクトルの次元数
        :param d_inner: 位置ごとのフィードフォワード層の中間次元数
        :param n_layers: エンコーダおよびデコーダの層数
        :param dropout: ドロップアウト確率
        """
        super(Transformer, self).__init__()

        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout

        # エンコーダの初期化
        self.encoder = Encoder(vocab_size=vocab_size,
                               positional_encoding=positional_encoding,
                               d_model=d_model,
                               n_heads=n_heads,
                               d_queries=d_queries,
                               d_values=d_values,
                               d_inner=d_inner,
                               n_layers=n_layers,
                               dropout=dropout)

        # デコーダの初期化
        self.decoder = Decoder(vocab_size=vocab_size,
                               positional_encoding=positional_encoding,
                               d_model=d_model,
                               n_heads=n_heads,
                               d_queries=d_queries,
                               d_values=d_values,
                               d_inner=d_inner,
                               n_layers=n_layers,
                               dropout=dropout)

        # 重みの初期化
        self.init_weights()

    def init_weights(self):
        """
        Transformerモデル内の重みを初期化する
        """
        # Glorot（Xavier）一様初期化（gain=1）を適用
        for p in self.parameters():
            # Glorot初期化はテンソルが2次元以上である必要がある
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.)

        # 埋め込み層と出力層の重みを共有する
        nn.init.normal_(self.encoder.embedding.weight, mean=0., std=math.pow(self.d_model, -0.5))
        self.decoder.embedding.weight = self.encoder.embedding.weight
        self.decoder.fc.weight = self.decoder.embedding.weight

        print("Model initialized.")

    def forward(self, encoder_sequences, decoder_sequences, encoder_sequence_lengths, decoder_sequence_lengths):
        """
        順伝播（フォワードプロパゲーション）

        :param encoder_sequences: ソース言語のシーケンス（テンソルサイズ: (N, エンコーダシーケンスのパッド長)）
        :param decoder_sequences: ターゲット言語のシーケンス（テンソルサイズ: (N, デコーダシーケンスのパッド長)）
        :param encoder_sequence_lengths: ソースシーケンスの実際の長さ（テンソルサイズ: (N)）
        :param decoder_sequence_lengths: ターゲットシーケンスの実際の長さ（テンソルサイズ: (N)）
        :return: デコードされたターゲット言語のシーケンス（テンソルサイズ: (N, デコーダシーケンスのパッド長, vocab_size)）
        """
        # エンコーダの出力を計算
        encoder_sequences = self.encoder(encoder_sequences, encoder_sequence_lengths)
        # 結果のサイズ: (N, エンコーダシーケンスのパッド長, d_model)

        # デコーダの出力（ロジット）を計算
        decoder_sequences = self.decoder(decoder_sequences, decoder_sequence_lengths, encoder_sequences,
                                         encoder_sequence_lengths)
        # 結果のサイズ: (N, デコーダシーケンスのパッド長, vocab_size)

        return decoder_sequences


class LabelSmoothedCE(torch.nn.Module):
    """
    ラベル平滑化付きクロスエントロピー損失
    （正則化手法の一種）

    詳細は "Rethinking the Inception Architecture for Computer Vision", https://arxiv.org/abs/1512.00567 を参照
    """

    def __init__(self, eps=0.1):
        """
        :param eps: 平滑化係数
        """
        super(LabelSmoothedCE, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets, lengths):
        """
        順伝播（フォワードプロパゲーション）

        :param inputs: デコードされたターゲット言語シーケンス（テンソルサイズ: (N, パッド長, vocab_size)）
        :param targets: 正解のターゲット言語シーケンス（テンソルサイズ: (N, パッド長)）
        :param lengths: これらのシーケンスの実際の長さ（パディングを無視するため、テンソルサイズ: (N)）
        :return: 平均ラベル平滑化付きクロスエントロピー損失（スカラー）
        """
        # パッド位置を除去して平坦化する
        inputs, _, _, _ = pack_padded_sequence(input=inputs,
                                               lengths=lengths.cpu(),  # "lengths"はCPU上にあることが期待される
                                               batch_first=True,
                                               enforce_sorted=False)  # 結果のサイズ: (sum(lengths), vocab_size)
        targets, _, _, _ = pack_padded_sequence(input=targets,
                                                lengths=lengths.cpu(),
                                                batch_first=True,
                                                enforce_sorted=False)  # 結果のサイズ: (sum(lengths))

        # 正解シーケンスに対する "平滑化された" one-hot ベクトルを作成
        target_vector = torch.zeros_like(inputs).scatter(dim=1, index=targets.unsqueeze(1),
                                                         value=1.).to(device)
        # 結果のサイズ: (sum(lengths), n_classes), one-hot表現
        target_vector = target_vector * (1. - self.eps) + self.eps / target_vector.size(1)
        # 結果のサイズ: (sum(lengths), n_classes), 平滑化されたone-hot表現

        # 平滑化付きクロスエントロピー損失を計算
        loss = (-1 * target_vector * F.log_softmax(inputs, dim=1)).sum(dim=1)  # 結果のサイズ: (sum(lengths))

        # 損失の平均値を計算
        loss = torch.mean(loss)

        return loss
