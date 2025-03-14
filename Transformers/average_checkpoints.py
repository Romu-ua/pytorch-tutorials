import torch
import os
from collections import OrderedDict

source_folder = "./"  # 平均化するチェックポイントが保存されているフォルダ
starts_with = "step"  # チェックポイントのファイル名の先頭にある文字列
ends_with = ".pth.tar"  # チェックポイントのファイル名の末尾の文字列

# チェックポイントのファイル名リストを取得
checkpoint_names = [f for f in os.listdir(source_folder) if f.startswith(starts_with) and f.endswith(ends_with)]
assert len(checkpoint_names) > 0, "チェックポイントが見つかりません！"

# チェックポイントからパラメータを平均化する
averaged_params = OrderedDict()
for c in checkpoint_names:
    checkpoint = torch.load(c)['model']  # 各チェックポイントのモデル部分を読み込む
    checkpoint_params = checkpoint.state_dict()  # パラメータの状態辞書を取得
    checkpoint_param_names = checkpoint_params.keys()
    for param_name in checkpoint_param_names:
        # 初回は初期化し、以降は平均値に加算
        if param_name not in averaged_params:
            averaged_params[param_name] = checkpoint_params[param_name].clone() * (1 / len(checkpoint_names))
        else:
            averaged_params[param_name] += checkpoint_params[param_name] * (1 / len(checkpoint_names))

# 平均化したパラメータを読み込むために、最初のチェックポイントをサンプルとして使用する
averaged_checkpoint = torch.load(checkpoint_names[0])['model']
for param_name in averaged_checkpoint.state_dict().keys():
    assert param_name in averaged_params
averaged_checkpoint.load_state_dict(averaged_params)

# 平均化したチェックポイントを保存する
torch.save({'model': averaged_checkpoint}, "averaged_transformer_checkpoint.pth.tar")
