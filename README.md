# Fine Chuning VGG16
本レポジトリは知能情報総合実験・データマイニング班 G2により作成されました。

## 環境構築
Poetryで依存関係をインストールすることをお勧めします。

下記のコマンドは、Python3.10以上の環境で実行する必要があります:
```bash
# 参照先: https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -

# Poetry経由で依存関係をインストール
poetry install
```

pipでも依存関係のインストールが可能です。
```bash
pip install -r requirements.txt
```

macOSで実行する場合には以下のコマンドを行ってください。
```bash
pip install tensorflow-macos==2.13.0
```

## データセットを準備
G2により、まとめたデータセットを使う場合は[datasets](https://gitlab.ie.u-ryukyu.ac.jp/e215742/datasets)からのインストールが可能です。
```bash
git clone https://gitlab.ie.u-ryukyu.ac.jp/e215742/datasets.git
```

また、このデータセットに限らず別のデータセットを用いても構いません。

## 実行方法

```bash
python3 fine_chuning_vgg16.py  --train-directory-path /path/to/train_dir --test-directory-path /path/to/test_dir
```
