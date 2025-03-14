import youtokentome
import codecs
import os
import torch
from random import shuffle
from itertools import groupby
from torch.nn.utils.rnn import pad_sequence


class SequenceLoader(object):
    """
    Transformerモデルにデータのバッチを読み込むためのイテレータ

    学習用の場合:
        各バッチは、約 tokens_in_batch 個のターゲット言語トークンを含み、
        パディングを最小限に抑えるために同じ長さのターゲット言語シーケンス、
        また、同様にパディングを最小限に抑えるために非常に近い（場合によっては同じ）長さのソース言語シーケンスを含む。
        バッチはシャッフルされる。

    検証・テストの場合:
        各バッチは、ファイルから読み込んだ順序と同じ順序で、
        単一のソース-ターゲットペアのみを含む。
    """

    def __init__(self, data_folder, source_suffix, target_suffix, split, tokens_in_batch):
        """
        :param data_folder: ソースおよびターゲット言語のデータファイルが格納されているフォルダ
        :param source_suffix: ソース言語ファイルのファイル名接尾辞
        :param target_suffix: ターゲット言語ファイルのファイル名接尾辞
        :param split: train, val, test のいずれか
        :param tokens_in_batch: 各バッチに含むターゲット言語トークン数
        """
        self.tokens_in_batch = tokens_in_batch
        self.source_suffix = source_suffix
        self.target_suffix = target_suffix
        assert split.lower() in {"train", "val", "test"}, "'split' は 'train', 'val', 'test' のいずれかである必要があります！（大文字小文字は区別しません）"
        self.split = split.lower()

        # これは学習用かどうかの判定
        self.for_training = self.split == "train"

        # BPEモデルを読み込む
        self.bpe_model = youtokentome.BPE(model=os.path.join(data_folder, "bpe.model"))

        # データの読み込み
        with codecs.open(os.path.join(data_folder, ".".join([split, source_suffix])), "r", encoding="utf-8") as f:
            source_data = f.read().split("\n")[:-1]
        with codecs.open(os.path.join(data_folder, ".".join([split, target_suffix])), "r", encoding="utf-8") as f:
            target_data = f.read().split("\n")[:-1]
        assert len(source_data) == len(target_data), "ソースシーケンスとターゲットシーケンスの数が異なります！"
        # 各シーケンスのBPEトークン数を計算（ソースは<BOS>/<EOS>無し、ターゲットは<BOS>と<EOS>あり）
        source_lengths = [len(s) for s in self.bpe_model.encode(source_data, bos=False, eos=False)]
        target_lengths = [len(t) for t in self.bpe_model.encode(target_data, bos=True, eos=True)]
        self.data = list(zip(source_data, target_data, source_lengths, target_lengths))

        # 学習用の場合、後で itertools.groupby() を使うためにターゲットシーケンスの長さで事前にソートする
        if self.for_training:
            self.data.sort(key=lambda x: x[3])

        # バッチの作成
        self.create_batches()

    def create_batches(self):
        """
        1エポック分のバッチを準備する
        """

        # 学習用の場合
        if self.for_training:
            # ターゲットシーケンスの長さに基づいてグループ化またはチャンク化する
            chunks = [list(g) for _, g in groupby(self.data, key=lambda x: x[3])]

            # 同じターゲットシーケンスの長さを持つバッチを作成する
            self.all_batches = list()
            for chunk in chunks:
                # チャンク内をソースシーケンスの長さでソートし、バッチ内のソースシーケンスも似た長さにする
                chunk.sort(key=lambda x: x[2])
                # このチャンクにおけるターゲットシーケンス長で、期待されるバッチサイズ（トークン数）を割って、各バッチに含むシーケンス数を計算する
                seqs_per_batch = self.tokens_in_batch // chunk[0][3]
                # チャンクをバッチに分割する
                self.all_batches.extend([chunk[i: i + seqs_per_batch] for i in range(0, len(chunk), seqs_per_batch)])

            # バッチをシャッフルする
            shuffle(self.all_batches)
            self.n_batches = len(self.all_batches)
            self.current_batch = -1
        else:
            # 検証・テストの場合は、単に1ペアずつ返す
            self.all_batches = [[d] for d in self.data]
            self.n_batches = len(self.all_batches)
            self.current_batch = -1

    def __iter__(self):
        """
        イテレータとして必要なメソッド
        """
        return self

    def __next__(self):
        """
        イテレータとして必要なメソッド

        :returns: 次のバッチ。バッチは以下の内容を含む：
            ソース言語シーケンス（テンソルサイズ: (N, エンコーダシーケンスのパッド長)）
            ターゲット言語シーケンス（テンソルサイズ: (N, デコーダシーケンスのパッド長)）
            正しいソースシーケンスの長さ（テンソルサイズ: (N)）
            正しいターゲットシーケンスの長さ（通常、バケット分けによりデコーダシーケンスのパッド長と同じ、テンソルサイズ: (N)）
        """
        # 現在のバッチインデックスを更新
        self.current_batch += 1
        try:
            source_data, target_data, source_lengths, target_lengths = zip(*self.all_batches[self.current_batch])
        # 全バッチをイテレーションしたらStopIterationを投げる
        except IndexError:
            raise StopIteration

        # BPEモデルを用いて、各シーケンスを単語ID（トークン）に変換する
        source_data = self.bpe_model.encode(source_data, output_type=youtokentome.OutputType.ID, bos=False, eos=False)
        target_data = self.bpe_model.encode(target_data, output_type=youtokentome.OutputType.ID, bos=True, eos=True)

        # ソースとターゲットシーケンスをパディング済みテンソルに変換する
        source_data = pad_sequence(sequences=[torch.LongTensor(s) for s in source_data],
                                   batch_first=True,
                                   padding_value=self.bpe_model.subword_to_id('<PAD>'))
        target_data = pad_sequence(sequences=[torch.LongTensor(t) for t in target_data],
                                   batch_first=True,
                                   padding_value=self.bpe_model.subword_to_id('<PAD>'))

        # 長さをテンソルに変換する
        source_lengths = torch.LongTensor(source_lengths)
        target_lengths = torch.LongTensor(target_lengths)

        return source_data, target_data, source_lengths, target_lengths
