import torch
import os
import wget
import tarfile
import shutil
import codecs
import youtokentome
import math
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_data(data_folder):
    """
    Downloads the training, validation, and test files for WMT '14 en-de translation task.

    Training: Europarl v7, Common Crawl, News Commentary v9
    Validation: newstest2013
    Testing: newstest2014

    The homepage for the WMT '14 translation task, https://www.statmt.org/wmt14/translation-task.html, contains links to
    the datasets.

    :param data_folder: the folder where the files will be downloaded

    """
    train_urls = ["http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
                  "https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
                  "http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz"]

    print("\n\nThis may take a while.")

    # tarファイル保存用のフォルダの作成
    if not os.path.isdir(os.path.join(data_folder, "tar files")):
        os.mkdir(os.path.join(data_folder, "tar files"))
    # 解凍用フォルダの作成と初期化
    if os.path.isdir(os.path.join(data_folder, "extracted files")):
        shutil.rmtree(os.path.join(data_folder, "extracted files"))
        os.mkdir(os.path.join(data_folder, "extracted files"))

    # 各URLごとにダウンロードと解凍の処理
    for url in train_urls:
        filename = url.split("/")[-1]
        if not os.path.exists(os.path.join(data_folder, "tar files", filename)):
            print("\nDownloading %s..." % filename)
            wget.download(url, os.path.join(data_folder, "tar files", filename))
        print("\nExtracting %s..." % filename)
        tar = tarfile.open(os.path.join(data_folder, "tar files", filename))
        members = [m for m in tar.getmembers() if "de-en" in m.path]
        tar.extractall(os.path.join(data_folder, "extracted files"), members=members)

    # searchBLEUを用いた検証・テストデータのダウンロード
    print("\n")
    os.system("sacrebleu -t wmt13 -l en-de --echo src > '" + os.path.join(data_folder, "val.en") + "'")
    os.system("sacrebleu -t wmt13 -l en-de --echo ref > '" + os.path.join(data_folder, "val.de") + "'")
    print("\n")
    os.system("sacrebleu -t wmt14/full -l en-de --echo src > '" + os.path.join(data_folder, "test.en") + "'")
    os.system("sacrebleu -t wmt14/full -l en-de --echo ref > '" + os.path.join(data_folder, "test.de") + "'")

   # 抽出されたファイルの整理
    for dir in [d for d in os.listdir(os.path.join(data_folder, "extracted files")) if
                os.path.isdir(os.path.join(data_folder, "extracted files", d))]:
        for f in os.listdir(os.path.join(data_folder, "extracted files", dir)):
            shutil.move(os.path.join(data_folder, "extracted files", dir, f),
                        os.path.join(data_folder, "extracted files"))
        os.rmdir(os.path.join(data_folder, "extracted files", dir))


def prepare_data(data_folder, euro_parl=True, common_crawl=True, news_commentary=True, min_length=3, max_length=100,
                 max_length_ratio=1.5, retain_case=True):
    """
    Filters and prepares the training data, trains a Byte-Pair Encoding (BPE) model.

    :param data_folder: the folder where the files were downloaded
    :param euro_parl: include the Europarl v7 dataset in the training data?
    :param common_crawl: include the Common Crawl dataset in the training data?
    :param news_commentary: include theNews Commentary v9 dataset in the training data?
    :param min_length: exclude sequence pairs where one or both are shorter than this minimum BPE length
    :param max_length: exclude sequence pairs where one or both are longer than this maximum BPE length
    :param max_length_ratio: exclude sequence pairs where one is much longer than the other
    :param retain_case: retain case?
    """
    # Read raw files and combine
    german = list()
    english = list()
    files = list()
    # 関数の引数によってどのデータセットを使うのか決める
    # デフォルトで全てTrue
    assert euro_parl or common_crawl or news_commentary, "Set at least one dataset to True!"
    if euro_parl:
        files.append("europarl-v7.de-en")
    if common_crawl:
        files.append("commoncrawl.de-en")
    if news_commentary:
        files.append("news-commentary-v9.de-en")
    print("\nReading extracted files and combining...")
    for file in files:
        with codecs.open(os.path.join(data_folder, "extracted files", file + ".de"), "r", encoding="utf-8") as f:
            if retain_case:
                german.extend(f.read().split("\n"))
            else:
                german.extend(f.read().lower().split("\n"))
        with codecs.open(os.path.join(data_folder, "extracted files", file + ".en"), "r", encoding="utf-8") as f:
            if retain_case:
                english.extend(f.read().split("\n"))
            else:
                english.extend(f.read().lower().split("\n"))
        assert len(english) == len(german)

    # メモリ解放のための書き出し
    print("\nWriting to single files...")
    with codecs.open(os.path.join(data_folder, "train.en"), "w", encoding="utf-8") as f:
        f.write("\n".join(english))
    with codecs.open(os.path.join(data_folder, "train.de"), "w", encoding="utf-8") as f:
        f.write("\n".join(german))
    with codecs.open(os.path.join(data_folder, "train.ende"), "w", encoding="utf-8") as f:
        f.write("\n".join(english + german))
    del english, german  # free some RAM

    # BPEモデルの学習と適用
    print("\nLearning BPE...")
    youtokentome.BPE.train(data=os.path.join(data_folder, "train.ende"), vocab_size=37000,
                           model=os.path.join(data_folder, "bpe.model"))

    # Load BPE model
    print("\nLoading BPE model...")
    bpe_model = youtokentome.BPE(model=os.path.join(data_folder, "bpe.model"))

    # Re-read English, German
    print("\nRe-reading single files...")
    with codecs.open(os.path.join(data_folder, "train.en"), "r", encoding="utf-8") as f:
        english = f.read().split("\n")
    with codecs.open(os.path.join(data_folder, "train.de"), "r", encoding="utf-8") as f:
        german = f.read().split("\n")

    # Filter
    # フィルタ条件
    # 最小長チェック、最大長チェック、長さ比チェック
    print("\nFiltering...")
    pairs = list()
    for en, de in tqdm(zip(english, german), total=len(english)):
        en_tok = bpe_model.encode(en, output_type=youtokentome.OutputType.ID)
        de_tok = bpe_model.encode(de, output_type=youtokentome.OutputType.ID)
        len_en_tok = len(en_tok)
        len_de_tok = len(de_tok)
        if min_length < len_en_tok < max_length and \
                min_length < len_de_tok < max_length and \
                1. / max_length_ratio <= len_de_tok / len_en_tok <= max_length_ratio:
            pairs.append((en, de))
        else:
            continue
    # フィルタ結果の表示
    print("\nNote: %.2f per cent of en-de pairs were filtered out based on sub-word sequence length limits." % (100. * (
            len(english) - len(pairs)) / len(english)))

    # 新たにフィルタ済みの文だけを書き込み
    english, german = zip(*pairs)
    print("\nRe-writing filtered sentences to single files...")
    os.remove(os.path.join(data_folder, "train.en"))
    os.remove(os.path.join(data_folder, "train.de"))
    os.remove(os.path.join(data_folder, "train.ende"))
    with codecs.open(os.path.join(data_folder, "train.en"), "w", encoding="utf-8") as f:
        f.write("\n".join(english))
    with codecs.open(os.path.join(data_folder, "train.de"), "w", encoding="utf-8") as f:
        f.write("\n".join(german))
    del english, german, bpe_model, pairs

    print("\n...DONE!\n")



def get_positional_encoding(d_model, max_length=100):
    """
    論文で定義された位置エンコーディングを計算する

    :param d_model: Transformerモデル全体で使用されるベクトルの次元数
    :param max_length: 位置エンコーディングを計算する最大シーケンス長
    :return: 位置エンコーディング（テンソルサイズ: (1, max_length, d_model)）
    """
    positional_encoding = torch.zeros((max_length, d_model))  # (max_length, d_model)
    for i in range(max_length):
        for j in range(d_model):
            if j % 2 == 0:
                positional_encoding[i, j] = math.sin(i / math.pow(10000, j / d_model))
            else:
                positional_encoding[i, j] = math.cos(i / math.pow(10000, (j - 1) / d_model))

    positional_encoding = positional_encoding.unsqueeze(0)  # (1, max_length, d_model)

    return positional_encoding


def get_lr(step, d_model, warmup_steps):
    """
    学習率スケジュール。
    この実装は論文での定義の2倍で、公式T2Tリポジトリで使われている。

    :param step: 現在の学習ステップ番号
    :param d_model: Transformerモデル全体で使用されるベクトルの次元数
    :param warmup_steps: 学習率が線形に増加するウォームアップステップ数（論文の値の2倍）
    :return: 更新された学習率
    """
    lr = 2. * math.pow(d_model, -0.5) * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))
    return lr


def save_checkpoint(epoch, model, optimizer, prefix=''):
    """
    チェックポイントを保存する関数。各保存は前の保存を上書きする。

    :param epoch: エポック番号（0-indexed）
    :param model: Transformerモデル
    :param optimizer: オプティマイザ
    :param prefix: チェックポイントのファイル名のプレフィックス
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = prefix + 'transformer_checkpoint.pth.tar'
    torch.save(state, filename)


def change_lr(optimizer, new_lr):
    """
    指定された新しい学習率に学習率を変更する

    :param optimizer: 学習率を変更するオプティマイザ
    :param new_lr: 新しい学習率
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class AverageMeter(object):
    """
    指標の最新値、平均値、合計、カウントを管理するクラス
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0    # 最新の値
        self.avg = 0    # 平均値
        self.sum = 0    # 合計値
        self.count = 0  # カウント

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
