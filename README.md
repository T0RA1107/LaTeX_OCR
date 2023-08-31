# LaTeX OCR
## 概要
数式画像をLaTeXのスクリプトへと変換するモデル

## 動作確認/学習
- im2markup/scripts/preprocessingを用いてデータの前処理を行う。(確認だけならim2markup/data)を用いれば良い。
  - 実行方法などの詳細はhttps://github.com/harvardnlp/im2markup/tree/master から確認されたし
- 前処理したデータのパスなどをconfig.yamlに書き込む
- main.pyを実行する(LaTeX_OCRディレクトリで実行する必要あり)

## TODO
- 手書き数式データの補充
- データ拡張

## 参照・引用元
data/im2markup 引用元
https://github.com/harvardnlp/im2markup/tree/master

im2latex-100k
https://zenodo.org/record/56198#.YJjuCGZKgox
