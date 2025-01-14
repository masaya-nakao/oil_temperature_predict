# Oil Temperature Prediction

このプロジェクトでは、SARIMAモデルを使用してオイル温度（OT）の予測を行います。`oil_temperature_prediction.py`スクリプトは、データの読み込み、前処理、モデルの訓練、予測、および評価を行います。

## ファイル構成

- `oil_temperature_prediction.py`: メインのスクリプトファイル。データの読み込み、前処理、SARIMAモデルの訓練と予測、評価を行います。
- `ett.csv`: 使用するデータセット。オイル温度（OT）と日付（date）が含まれています。

## 必要なライブラリ

以下のPythonライブラリが必要です：

- pandas
- matplotlib
- statsmodels
- scikit-learn

これらのライブラリは、以下のコマンドでインストールできます：

```bash
pip install pandas matplotlib statsmodels scikit-learn

## 実行方法
必要なライブラリをインストールします。
ett.csvファイルを同じディレクトリに配置します。
oil_temperature_prediction.pyスクリプトを実行します。
python oil_temperature_prediction.py