import torch
import torch.nn.functional as F
import youtokentome
import math

# デバイスの設定（GPUが利用可能ならCUDA、なければCPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BPEモデルの読み込み
bpe_model = youtokentome.BPE(model="/media/ssd/transformer data/bpe.model")

# Transformerモデルの読み込み（チェックポイントから）
checkpoint = torch.load("averaged_transformer_checkpoint.pth.tar")
model = checkpoint['model'].to(device)
model.eval()  # 評価モードに設定

def translate(source_sequence, beam_size=4, length_norm_coefficient=0.6):
    """
    ソース言語のシーケンスをターゲット言語に翻訳する（ビームサーチによるデコード）

    :param source_sequence: ソース言語のシーケンス（文字列またはBPEインデックスのテンソル）
    :param beam_size: ビームサイズ
    :param length_norm_coefficient: デコードされたシーケンスのスコアを長さで正規化するための係数
    :return: 最良の仮説と、全候補仮説のリスト
    """
    with torch.no_grad():
        # ビームサイズの設定
        k = beam_size

        # 完成仮説の最小数（ビームサイズと10のうち小さい方）
        n_completed_hypotheses = min(k, 10)

        # 語彙数の取得
        vocab_size = bpe_model.vocab_size()

        # ソースシーケンスが文字列の場合、BPEインデックスのテンソルに変換する
        if isinstance(source_sequence, str):
            encoder_sequences = bpe_model.encode(source_sequence,
                                                 output_type=youtokentome.OutputType.ID,
                                                 bos=False,
                                                 eos=False)
            encoder_sequences = torch.LongTensor(encoder_sequences).unsqueeze(0)  # サイズ: (1, ソースシーケンス長)
        else:
            encoder_sequences = source_sequence
        encoder_sequences = encoder_sequences.to(device)  # (1, ソースシーケンス長)
        encoder_sequence_lengths = torch.LongTensor([encoder_sequences.size(1)]).to(device)  # (1)

        # エンコーダ部の計算（ソースシーケンスのエンコード）
        encoder_sequences = model.encoder(encoder_sequences=encoder_sequences,
                                          encoder_sequence_lengths=encoder_sequence_lengths)
        # 結果のサイズ: (1, ソースシーケンス長, d_model)

        # 初期の仮説は<BOS>トークンのみ
        hypotheses = torch.LongTensor([[bpe_model.subword_to_id('<BOS>')]]).to(device)  # サイズ: (1, 1)
        hypotheses_lengths = torch.LongTensor([hypotheses.size(1)]).to(device)  # サイズ: (1)

        # 仮説のスコアを保存するテンソル（初期は0）
        hypotheses_scores = torch.zeros(1).to(device)  # サイズ: (1)

        # 完成した仮説とそのスコアを格納するリスト
        completed_hypotheses = list()
        completed_hypotheses_scores = list()

        # デコード開始
        step = 1

        # 現在の不完全な仮説の数（s）はビームサイズ以下。最初は<BOS>のみなので s=1
        while True:
            s = hypotheses.size(0)
            # デコーダ部の計算
            decoder_sequences = model.decoder(decoder_sequences=hypotheses,
                                              decoder_sequence_lengths=hypotheses_lengths,
                                              encoder_sequences=encoder_sequences.repeat(s, 1, 1),
                                              encoder_sequence_lengths=encoder_sequence_lengths.repeat(s))
            # 結果のサイズ: (s, step, vocab_size)

            # 現ステップのスコアを取得（各仮説の最後のトークンの出力）
            scores = decoder_sequences[:, -1, :]  # サイズ: (s, vocab_size)
            scores = F.log_softmax(scores, dim=-1)  # サイズ: (s, vocab_size)

            # 直前のステップのスコアに現在のスコアを加えて、すべての新しい仮説のスコアを計算する
            scores = hypotheses_scores.unsqueeze(1) + scores  # サイズ: (s, vocab_size)

            # 全スコアを展開し、上位 k 個のスコアとそのインデックスを取得する
            top_k_hypotheses_scores, unrolled_indices = scores.view(-1).topk(k, 0, True, True)  # サイズ: (k)

            # 展開されたインデックスから、元のスコアテンソルのインデックスに変換
            prev_word_indices = unrolled_indices // vocab_size  # (k)
            next_word_indices = unrolled_indices % vocab_size  # (k)

            # これらのインデックスを用いて新たな上位 k 仮説を構築する
            top_k_hypotheses = torch.cat([hypotheses[prev_word_indices], next_word_indices.unsqueeze(1)],
                                         dim=1)  # サイズ: (k, step + 1)

            # 新しい仮説のうち、どれが完了（<EOS>に到達）しているか判定する
            complete = next_word_indices == bpe_model.subword_to_id('<EOS>')  # サイズ: (k), bool

            # 完成した仮説とそのスコア（長さで正規化したもの）を保存する
            completed_hypotheses.extend(top_k_hypotheses[complete].tolist())
            norm = math.pow(((5 + step) / (5 + 1)), length_norm_coefficient)
            completed_hypotheses_scores.extend((top_k_hypotheses_scores[complete] / norm).tolist())

            # 十分な数の完成仮説が得られたらループを終了する
            if len(completed_hypotheses) >= n_completed_hypotheses:
                break

            # まだ完成していない仮説で続行する
            hypotheses = top_k_hypotheses[~complete]  # サイズ: (s, step + 1)
            hypotheses_scores = top_k_hypotheses_scores[~complete]  # サイズ: (s)
            hypotheses_lengths = torch.LongTensor(hypotheses.size(0) * [hypotheses.size(1)]).to(device)  # サイズ: (s)

            # 長すぎる場合はループを終了する
            if step > 100:
                break
            step += 1

        # 完成した仮説が一つもない場合、未完成の仮説を利用する
        if len(completed_hypotheses) == 0:
            completed_hypotheses = hypotheses.tolist()
            completed_hypotheses_scores = hypotheses_scores.tolist()

        # 仮説をデコードする
        all_hypotheses = list()
        for i, h in enumerate(bpe_model.decode(completed_hypotheses, ignore_ids=[0, 2, 3])):
            all_hypotheses.append({"hypothesis": h, "score": completed_hypotheses_scores[i]})

        # 最もスコアが高い完成仮説を選択する
        i = completed_hypotheses_scores.index(max(completed_hypotheses_scores))
        best_hypothesis = all_hypotheses[i]["hypothesis"]

        return best_hypothesis, all_hypotheses

if __name__ == '__main__':
    translate("Anyone who retains the ability to recognise beauty will never become old.")
