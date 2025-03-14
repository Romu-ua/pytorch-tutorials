import torch
import sacrebleu
from translate import translate
from tqdm import tqdm
from dataloader import SequenceLoader
import youtokentome
import codecs
import os

# Python内でsacreBLEUを使用するか、コマンドラインで使用するかの選択
# Python内で使用すると、prepare_data.pyでダウンロードされたテストデータが使われる
# コマンドラインで使用すると、sacreBLEUによって自動でダウンロードされるテストデータが使われる...
# ...そして、使用された正確なBLEUの計算方法を示す標準的な署名が出力される（他者が再現・比較できるように重要）
sacrebleu_in_python = False

# translate.py内で正しいモデルチェックポイントが選択されていることを確認する

# データローダーの作成
test_loader = SequenceLoader(data_folder="/media/ssd/transformer data",
                             source_suffix="en",
                             target_suffix="de",
                             split="test",
                             tokens_in_batch=None)
test_loader.create_batches()

# 評価
with torch.no_grad():
    hypotheses = list()
    references = list()
    for i, (source_sequence, target_sequence, source_sequence_length, target_sequence_length) in enumerate(
            tqdm(test_loader, total=test_loader.n_batches)):
        # 各ソースシーケンスに対して翻訳を実行し、最良の仮説をリストに追加
        hypotheses.append(translate(source_sequence=source_sequence,
                                    beam_size=4,
                                    length_norm_coefficient=0.6)[0])
        # ターゲットシーケンスをデコードして参照文リストに追加
        references.extend(test_loader.bpe_model.decode(target_sequence.tolist(), ignore_ids=[0, 2, 3]))
    if sacrebleu_in_python:
        # Python内でsacreBLEUを使用してBLEUスコアを計算し、各トークナイゼーションオプションの結果を表示
        print("\n13a tokenization, cased:\n")
        print(sacrebleu.corpus_bleu(hypotheses, [references]))
        print("\n13a tokenization, caseless:\n")
        print(sacrebleu.corpus_bleu(hypotheses, [references], lowercase=True))
        print("\nInternational tokenization, cased:\n")
        print(sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl'))
        print("\nInternational tokenization, caseless:\n")
        print(sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl', lowercase=True))
        print("\n")
    else:
        # もしPython内で計算しない場合、翻訳結果をファイルに保存してコマンドラインでsacreBLEUを実行する
        with codecs.open("translated_test.de", "w", encoding="utf-8") as f:
            f.write("\n".join(hypotheses))
        print("\n13a tokenization, cased:\n")
        os.system("cat translated_test.de | sacrebleu -t wmt14/full -l en-de")
        print("\n13a tokenization, caseless:\n")
        os.system("cat translated_test.de | sacrebleu -t wmt14/full -l en-de -lc")
        print("\nInternational tokenization, cased:\n")
        os.system("cat translated_test.de | sacrebleu -t wmt14/full -l en-de -tok intl")
        print("\nInternational tokenization, caseless:\n")
        os.system("cat translated_test.de | sacrebleu -t wmt14/full -l en-de -tok intl -lc")
        print("\n")
    print(
        "The first value (13a tokenization, cased) is how the BLEU score is officially calculated by WMT (mteval-v13a.pl). \n"
        "This is probably not how it is calculated in the 'Attention Is All You Need' paper, however.\n"
        "See https://github.com/tensorflow/tensor2tensor/issues/317#issuecomment-380970191 for more details.\n")
