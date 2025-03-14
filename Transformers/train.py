import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import time
from model import Transformer, LabelSmoothedCE
from dataloader import SequenceLoader
from utils import *

data_folder = './transformer_data'

# モデルのパラメータ
d_model = 512  # Transformerモデル全体で使用されるベクトルの次元数
n_heads = 8  # マルチヘッド注意機構のヘッド数
d_queries = 64  # マルチヘッド注意機構におけるクエリ（およびキー）のベクトルの次元数
d_values = 64  # マルチヘッド注意機構におけるバリューのベクトルの次元数
d_inner = 2048  # 位置ごとのフィードフォワード層の中間次元数
n_layers = 6  # エンコーダおよびデコーダの層数
dropout = 0.1  # ドロップアウトの確率
positional_encoding = get_positional_encoding(d_model=d_model,
                                              max_length=160)  # 最大パッド長までの位置エンコーディング

# 学習のパラメータ
checkpoint = 'transformer_checkpoint.pth.tar'  # モデルチェックポイントのパス（チェックポイントがなければNone）
tokens_in_batch = 2000  # バッチ内のターゲット言語のトークン数（バッチサイズ）
batches_per_step = 25000 // tokens_in_batch  # 何バッチごとにパラメータ更新（1ステップ）を行うか
print_frequency = 20  # 何ステップごとに状態を出力するか
n_steps = 100000  # 総学習ステップ数
warmup_steps = 8000  # 学習率を線形に増加させるウォームアップステップ数（論文の値の2倍；公式Transformerリポジトリと同様）
step = 1  # 現在のステップ番号（0では後の計算でエラーになるため1から開始）
lr = get_lr(step=step, d_model=d_model,
            warmup_steps=warmup_steps)  # 学習率スケジュールを取得；詳細はutils.pyを参照。論文のスケジュールの2倍になっている
start_epoch = 0  # 開始エポック番号
betas = (0.9, 0.98)  # Adamオプティマイザのβ係数
epsilon = 1e-9  # Adamオプティマイザのイプシロン項
label_smoothing = 0.1  # クロスエントロピー損失におけるラベル平滑化係数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPUが利用可能ならCUDAを、なければCPUを使用（CPUは実用的ではないため）
cudnn.benchmark = False  # 入力テンソルサイズが可変なため、cuDNNのベンチマークは無効にする

def main():
    """
    学習と検証を行うメイン関数
    """
    global checkpoint, step, start_epoch, epoch, epochs

    # データローダーの初期化
    train_loader = SequenceLoader(data_folder="./transformer_data",
                                  source_suffix="en",
                                  target_suffix="de",
                                  split="train",
                                  tokens_in_batch=tokens_in_batch)
    val_loader = SequenceLoader(data_folder="./transformer_data",
                                source_suffix="en",
                                target_suffix="de",
                                split="val",
                                tokens_in_batch=tokens_in_batch)

    # モデルの初期化またはチェックポイントからの読み込み
    if checkpoint is None:
        model = Transformer(vocab_size=train_loader.bpe_model.vocab_size(),
                            positional_encoding=positional_encoding,
                            d_model=d_model,
                            n_heads=n_heads,
                            d_queries=d_queries,
                            d_values=d_values,
                            d_inner=d_inner,
                            n_layers=n_layers,
                            dropout=dropout)
        optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad],
                                     lr=lr,
                                     betas=betas,
                                     eps=epsilon)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nチェックポイントをエポック %d から読み込みました。\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # 損失関数の定義（ラベル平滑化付きクロスエントロピー）
    criterion = LabelSmoothedCE(eps=label_smoothing)

    # モデルと損失関数をデフォルトデバイス（GPUまたはCPU）へ移動
    model = model.to(device)
    criterion = criterion.to(device)

    # 学習する総エポック数の計算
    epochs = (n_steps // (train_loader.n_batches // batches_per_step)) + 1

    # 各エポックごとの処理
    for epoch in range(start_epoch, epochs):
        # 現在のステップ番号の更新
        step = epoch * train_loader.n_batches // batches_per_step

        # 1エポック分の学習処理
        train_loader.create_batches()
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              step=step)

        # 1エポック分の検証処理
        val_loader.create_batches()
        validate(val_loader=val_loader,
                 model=model,
                 criterion=criterion)

        # チェックポイントの保存
        save_checkpoint(epoch, model, optimizer)



def train(train_loader, model, criterion, optimizer, epoch, step):
    """
    1エポック分の学習を行う関数

    :param train_loader: 学習データのローダー
    :param model: モデル
    :param criterion: ラベル平滑化付きクロスエントロピー損失
    :param optimizer: オプティマイザ
    :param epoch: エポック番号
    """
    model.train()  # 学習モードに設定（これによりドロップアウトなどが有効になる）

    # メトリクスの追跡
    data_time = AverageMeter()  # データ読み込み時間の計測
    step_time = AverageMeter()  # 順伝播＋逆伝播にかかる時間の計測
    losses = AverageMeter()  # 損失の計測

    # 開始時間の記録
    start_data_time = time.time()
    start_step_time = time.time()

    # バッチ処理
    for i, (source_sequences, target_sequences, source_sequence_lengths, target_sequence_lengths) in enumerate(train_loader):

        # デフォルトデバイス（GPUまたはCPU）へ移動
        source_sequences = source_sequences.to(device)  # (N, 現バッチ内の最大ソースシーケンス長)
        target_sequences = target_sequences.to(device)  # (N, 現バッチ内の最大ターゲットシーケンス長)
        source_sequence_lengths = source_sequence_lengths.to(device)  # (N)
        target_sequence_lengths = target_sequence_lengths.to(device)  # (N)

        # データの読み込みにかかった時間を更新
        data_time.update(time.time() - start_data_time)

        # 順伝播
        predicted_sequences = model(source_sequences, target_sequences, source_sequence_lengths,
                                    target_sequence_lengths)  # (N, 現バッチ内の最大ターゲットシーケンス長, 語彙数)

        # 注意:
        # ターゲットシーケンスが "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> ..." の場合、
        # <BOS> は予測されないので "w1 w2 ... wN <EOS>" の部分のみを対象とする必要がある。
        # そのため、パディングは (length - 1) 以降の位置から開始する。
        loss = criterion(inputs=predicted_sequences,
                         targets=target_sequences[:, 1:],
                         lengths=target_sequence_lengths - 1)  # スカラー値の損失

        # 逆伝播
        (loss / batches_per_step).backward()

        # 損失の更新（トークン数に基づく重み付け）
        losses.update(loss.item(), (target_sequence_lengths - 1).sum().item())

        # batches_per_step バッチ分の勾配が蓄積されたら、モデルを更新する
        if (i + 1) % batches_per_step == 0:
            optimizer.step()
            optimizer.zero_grad()

            # ステップ数を更新
            step += 1

            # 各ステップごとに学習率の更新
            change_lr(optimizer, new_lr=get_lr(step=step, d_model=d_model, warmup_steps=warmup_steps))

            # このステップにかかった時間を更新
            step_time.update(time.time() - start_step_time)

            # print_frequency ステップごとに状況を出力する
            if step % print_frequency == 0:
                print('Epoch {0}/{1}-----'
                      'Batch {2}/{3}-----'
                      'Step {4}/{5}-----'
                      'Data Time {data_time.val:.3f} ({data_time.avg:.3f})-----'
                      'Step Time {step_time.val:.3f} ({step_time.avg:.3f})-----'
                      'Loss {losses.val:.4f} ({losses.avg:.4f})'.format(
                          epoch + 1, epochs,
                          i + 1, train_loader.n_batches,
                          step, n_steps,
                          step_time=step_time,
                          data_time=data_time,
                          losses=losses))

            # ステップ時間の計測をリセット
            start_step_time = time.time()

            # 最後の1,2エポックの場合、定期的にチェックポイントを保存（平均化のため）
            if epoch in [epochs - 1, epochs - 2] and step % 1500 == 0:  # 'epoch' は0-indexed
                save_checkpoint(epoch, model, optimizer, prefix='step' + str(step) + "_")

        # 次のバッチ読み込み前に、データ読み込み時間の計測をリセット
        start_data_time = time.time()


def validate(val_loader, model, criterion):
    """
    1エポック分の検証を行う関数

    :param val_loader: 検証データのローダー
    :param model: モデル
    :param criterion: ラベル平滑化付きクロスエントロピー損失
    """
    model.eval()  # 評価モードに設定（これによりドロップアウトなどが無効になる）

    # 勾配計算を禁止する
    with torch.no_grad():
        losses = AverageMeter()  # 損失の計測用
        # バッチ処理
        for i, (source_sequence, target_sequence, source_sequence_length, target_sequence_length) in enumerate(
                tqdm(val_loader, total=val_loader.n_batches)):
            # 各テンソルをデフォルトデバイスに移動
            source_sequence = source_sequence.to(device)  # (1, ソースシーケンス長)
            target_sequence = target_sequence.to(device)  # (1, ターゲットシーケンス長)
            source_sequence_length = source_sequence_length.to(device)  # (1)
            target_sequence_length = target_sequence_length.to(device)  # (1)

            # 順伝播
            predicted_sequence = model(source_sequence, target_sequence, source_sequence_length,
                                       target_sequence_length)  # (1, ターゲットシーケンス長, 語彙数)

            # 注意:
            # ターゲットシーケンスが "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> ..." の場合、
            # <BOS> は予測対象外なので、"w1 w2 ... wN <EOS>" の部分のみを対象とする。
            # そのため、パディングは (length - 1) の位置から開始する。
            loss = criterion(inputs=predicted_sequence,
                             targets=target_sequence[:, 1:],
                             lengths=target_sequence_length - 1)  # スカラーの損失値

            # 損失を記録する
            losses.update(loss.item(), (target_sequence_length - 1).sum().item())

        # 検証損失を出力
        print("\nValidation loss: %.3f\n\n" % losses.avg)


if __name__ == '__main__':
    main()
