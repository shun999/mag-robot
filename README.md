# インゴット異常検知 — DINOv3 + SVM

Jetson Orin Nano上でUSBカメラの画像を収集し、正常品・異常品の二値分類SVMを学習します。
学習したモデルでカメラ映像を判定し、Jetsonに接続した画面に表示します。

```text
USBカメラ → 背景差分でインゴット抽出 → 余白付き切り出し
         → 縦横比を保って正方形化・リサイズ → BGRからRGB・正規化
         → 凍結DINOv3（CLS＋パッチ平均＋パッチ最大）
         → StandardScaler → RBF SVM → 正常 / 異常
```

固定カメラ、無地背景、検査範囲に1個のインゴットを前提とします。インゴットの位置は
範囲内で変えられます。UR3への通信、Docker化、欠陥位置のヒートマップは含みません。

## 1. 環境を用意する

Ubuntu Arm64、Python 3.10〜3.12、GUI対応OpenCVを使用します。撮影・判定はJetsonの
デスクトップ端末から実行してください。Remote SSHのみではOpenCVの画面を表示できません。

[uv公式の導入手順](https://docs.astral.sh/uv/getting-started/installation/)でuvを導入した後、
リポジトリ直下で実行します。

```bash
uv sync --locked
```

### Jetson用PyTorch

PyTorchはJetPack / CUDA / Pythonに依存するため、プロジェクトの通常依存には含めていません。
`uv sync`後に、[NVIDIA公式のJetson用PyTorch導入手順](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
から、実機に対応するCUDA対応wheelと必要なシステムライブラリを導入してください。
JetPackの確認には `/etc/nv_tegra_release` と `dpkg-query -W nvidia-jetpack` を使います。
メタパッケージがない場合はL4T版だけからwheelを決めず、NVIDIAの互換情報を確認してください。

取得したwheelは次のように仮想環境へ導入します。ファイル名は実際のものに置き換えます。

```bash
uv pip install --python .venv/bin/python /absolute/path/to/jetson-compatible-torch.whl
```

本アプリの前処理はOpenCV / NumPyを使うため、torchvisionは不要です。
PyTorchとその追加依存は`uv.lock`の管理外です。導入後に`uv sync`を再実行すると
管理外パッケージを削除するため、以後は`uv run --no-sync`を使用してください。
アプリ依存を更新する場合は`uv sync --inexact`で管理外のPyTorchを保持します。
wheelの入手先と版、`uv pip freeze --python .venv/bin/python`の結果は実機ごとに保存してください。

### DINOv3コードと重み

[DINOv3公式リポジトリ](https://github.com/facebookresearch/dinov3)のコードを指定コミットに固定します。
学習用依存全体のインストールは不要です。変更のないGitチェックアウトを使用してください。

```bash
git clone https://github.com/facebookresearch/dinov3.git vendor/dinov3
git -C vendor/dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
```

公式ページの利用条件に同意してモデルへのアクセスを申請し、**ViT-S/16、LVD-1689M**の
重み`dinov3_vits16_pretrain_lvd1689m-08c60483.pth`を取得して`weights/`に配置します。
重みのダウンロードや利用条件への同意をアプリが自動実行することはありません。
既定以外の場所なら`train` / `infer` / `doctor`の`--repo`と`--weights`で指定します。
モデルを実行する際の自動ダウンロードはありません。

```bash
uv run --no-sync ingot doctor
```

カメラを接続し、画面が利用できる状態で実行します。`doctor`は依存・CUDA計算・カメラの
読取・DINOv3コード・重みの存在を個別に確認し、失敗があれば終了コード1を返します。
`--load-model`を付けると、他の確認がすべて成功した後、DINOv3の順伝播も実行します。
GUIの診断はビルドと画面環境変数の確認で、実際のウィンドウ表示は撮影時に確認してください。

GPUを使用しない動作確認には`doctor` / `train` / `infer`に`--device cpu`を付けます。
CPUの場合もDINOv3を実行できるPyTorchが必要です。既定のCUDA指定からCPUへ自動変更はしません。

## 2. 背景と検査範囲を調整する

`config.example.json`を必要に応じてコピーし、カメラ番号、解像度、FPSを変更します。
`camera.source`には`0`などの番号か`/dev/video0`、`/dev/v4l/by-id/...`を指定できます。
カメラが設定した解像度を返さない場合は停止するため、対応する解像度を指定してください。
FPSはカメラへの要求値で、実際の取得速度・判定速度の保証値ではありません。

```bash
uv run --no-sync ingot calibrate --config config.example.json --output calibration/default
```

1. インゴットを取り除き、照明とカメラを固定します。
2. 空の背景映像から検査範囲をドラッグし、Enterで確定します。Escは中止です。
3. 背景15フレームの中央値が自動取得されます。
4. インゴットを置き、`threshold`と`min_area`を調整します。
5. 抽出枠・切り出し画像・差分マスクを確認し、`s`で保存します。`q`で中止します。

`calibration/default/`に`config.json`と`background.png`を保存します。
既存の調整は上書きしません。カメラ、照明、背景、撮影解像度、検査範囲を変えたら
新しい保存先で再調整し、新しい撮影データ・モデルを作成します。

差分はBGR各チャンネルの絶対差の最大値で計算し、閾値処理とモルフォロジーで
抽出します。`kernel_size`はノイズ除去用の奇数、`margin`は切り出し周囲の余白率です。
抽出輪郭で画像をマスクしないため、インゴット表面と輪郭を含む矩形画像を保持します。
背景と似た色、影、反射、照明変動で抽出が崩れる場合は、背景や照明を調整してください。

次の場合はSVMに渡しません。

- 有効領域がない: `NO OBJECT`
- 最小面積以上の領域が複数ある: `INVALID`
- 検査範囲の端に領域が接触する: `INVALID`

接触した複数個体が一つの輪郭になった場合は識別できません。1個ずつ置いてください。

## 3. 正常品・異常品を撮影する

```bash
uv run --no-sync ingot capture --config calibration/default/config.json \
  --data data/ingots --item-id normal-001 --label normal

uv run --no-sync ingot capture --config calibration/default/config.json \
  --data data/ingots --item-id anomaly-001 --label anomaly
```

ウィンドウで**Space**を押すと現在の原画像と切り出し画像を保存します。`q`またはEscで終了します。
判定不可のフレームは保存しません。同じ個体を撮る間は同じIDを使い、別個体に交換したら
コマンドを終了して新しい`--item-id`で起動してください。IDは英数字から始まる
英数字・`_`・`-`・`.`の80文字以内です。正常品／異常品のラベルは人が判断して付けます。

**最低でも正常5個体・異常5個体が必要です。** 各5個体は動作確認の下限であり、
実用精度を保証する数ではありません。画像枚数だけ増やしても個体数の代わりにはなりません。
形状と表面の異常をそれぞれ含め、位置・向きなど実運用で起きる変動を撮影してください。
小さい傷が十分な画素数で写る距離、ピント、照明を調整します。

```text
data/ingots/samples/<sample_id>/
  raw.png       原画像（再学習時に領域抽出を再実行）
  crop.png      撮影時の切り出し画像（目視確認用）
  record.json   個体ID、ラベル、UTC撮影時刻、座標、設定、原画像ハッシュ
```

撮影はサンプルごとに一時ディレクトリへ保存した後に確定します。同じ個体IDに異なるラベル、
異なる調整設定の同一データディレクトリへの追加を拒否します。撮影・学習は同じデータに対し
同時実行しないでください。撮影後は原画像やメタデータを編集しないでください。

## 4. SVMを学習・評価する

```bash
uv run --no-sync ingot train --config calibration/default/config.json \
  --data data/ingots --output models/ingot-v1
```

- 原画像から共通コードで領域を再抽出し、DINOv3の1152次元特徴を計算します。
- 縦横比を維持する黒のパディング後、既定256×256へリサイズし、RGB・0〜1に変換します。
  正規化の平均は`(0.485, 0.456, 0.406)`、標準偏差は`(0.229, 0.224, 0.225)`です。
- 正常=0、異常=1。DINOv3は凍結し、CLS特徴・パッチ平均・パッチ最大を連結します。
- 各クラスを**個体単位**で学習60%・検証20%・テスト20%に分割します。
  検証・テスト個体数はそれぞれ`max(1, floor(個体数×0.2))`、残りが学習です。
  同じ個体の画像が分割をまたぐことはありません。既定seedは42で、`--seed`で変更できます。
- 学習データだけでStandardScalerと`SVC(kernel="rbf", class_weight="balanced")`を学習します。
  `C=[0.1,1,10]`、`gamma=["scale",0.001,0.01]`の9候補を比較します。
- 検証のbalanced accuracy、異常再現率の順に最大の候補を選択します。
  同点は上記の定義順です。選択したモデルをそのまま保存し、テストで一度評価します。
  検証・テストを含めた再学習はしません。
- 設定・原画像の変更、同一原画像の重複、ラベル矛盾、個体数不足はエラーにします。

```text
models/ingot-v1/
  svm.joblib       StandardScalerとSVM
  config.json      学習時の撮影・抽出・前処理設定
  background.png   学習時の背景
  metadata.json    ラベル、特徴仕様、重みSHA256、DINOv3コミット、ライブラリ版
  split.json      学習・検証・テストの個体IDとサンプルID
  report.json     選択候補、混同行列、異常の再現率・適合率・F1、正常の誤検出率、件数
```

混同行列は行=正解、列=予測で、順序は`[normal, anomaly]`です。指標は画像単位です。
同じ個体の類似画像を大量に撮ると、その個体が指標に強く影響します。
既存モデルの上書きは拒否します。更新時は`models/ingot-v2`など別の出力先を指定します。
実測のFPSや精度はカメラ・欠陥・データ量に依存するため、合格値は実画像を使って決めます。
テスト結果を見て設定を調整した場合は、最終確認用に未使用の個体を追加で用意してください。

## 5. リアルタイム判定する

```bash
uv run --no-sync ingot infer --model models/ingot-v1
```

原映像に抽出枠、`NORMAL / ANOMALY / NO OBJECT / INVALID`、異常スコア、1フレームの
読取・処理時間を表示し、別ウィンドウに切り出し画像を表示します。`q`またはEscで終了します。
異常スコアはSVMの`decision_function`で、正が異常側、負が正常側です。**確率ではありません。**
判定にはSVMの`predict`を使うため、境界付近ではスコアの丸め表示と見え方が異なる場合があります。

学習時の設定・背景をモデルから読み込みます。DINOv3重み、コード、前処理、特徴次元、
scikit-learn版の不一致やモデルファイルの変更を検出して停止します。
`joblib`は信頼できる自分のモデルだけを読み込んでください。
カメラ切断などの処理失敗時には表示をエラー画面に切り替えて終了し、前の判定を残しません。

再起動でカメラ番号だけが変わった場合には`--camera /dev/video2`などで変更できます。
物理的なカメラや撮影条件を変えた場合は、このオプションで流用せず再調整してください。

入力サイズは調整前の設定で`features.input_size`を16の倍数に変更できます。
大きくすると細部を保持しやすくなりますが計算量が増えます。変更後はデータ収集・学習を
やり直します。この初期版は微細欠陥や未学習の異常の検出を保証せず、写っていない裏面の検査はできません。

## テストと実機確認

PyTorch導入前は次のコマンドで検証できます。導入後は`--no-sync`で実行してください。

```bash
uv run --no-sync pytest
uv run --no-sync ruff check .
```

自動テストはカメラ・GPU・重みを使いません。領域抽出、前処理、個体分割、SVM学習、
保存・再読込、設定不一致、読取失敗時のリソース解放を検証します。
DINOv3部分はテスト用の特徴抽出器に置換し、画像処理・scikit-learnは実際の実装を実行します。

実機では以下を確認してください。

1. `doctor --load-model`が成功する。
2. 背景だけ、正常品、異常品、複数個体、検査範囲端への接触で抽出表示が適切になる。
3. 各クラス5個体以上を撮影し、学習とレポート出力が成功する。
4. アプリを再起動し、保存済みモデルから正常・異常をライブ判定できる。
5. カメラを外すと古い判定が消え、エラー終了する。

データ、背景、モデル、DINOv3コード、重みはGit管理対象外です。必要なバックアップは別途行ってください。
