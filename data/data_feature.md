# dataの構成
参照元
https://zenodo.org/record/56198#.YJjuCGZKgox

- formula_images
    - 数式の画像
    - 黒い文字に透過背景となっている
    - 中身を見るときは、背景を白くすることを推奨
- im2latex_formulas.lst
    - 0-indexedの数式idでソートされた数式のLaTeX表現が書かれている
    - そのまま`open("im2latex_formulas.lst").read()`などすると`UnicodeDecodeError`が出る
        - Errorが出るものは`wierds.txt`にDecodeする前のバイト文字列として記録している
        - Errorが出るデータは31個
        - これらには`\xe7`,`\xa5`,`\x95`,`\xa0`...などのように、16進数表式:`\xnn`=$\rm{(nn)_{16}}$が含まれている(謎)
        - これらは単純に空白に変えても良さそう(見た目は一致している)
    - コメントアウト%がちょこちょこ含まれている
        - 上記のErrorが出るもの以外でコメントアウトを含むものは`include_comment_out.txt`に記録している
        - コメントアウトを含んでいるものは3467個
        - 1つの数式に%は最大60個含まれている
        - ルールは不明だが、%の後ろは必ずしも消えているわけではない
        - id87606(画像:7605bfe3a8.png)のようにコメントアウトで数式が消されて画像が空になっているものや、コメントアウトされているはずなのに数式が表示されているものがあり、ややこしいので捨ててしまうのが無難
- im2latex_{test | train | validate}.lst
    - test, train, validateに分かれた3つのLSTファイル
    - 数式id, 数式画像名, レンダリング方式(全部basicだけど)が1行ごとに書かれている

`data/readme.txt`も参照
