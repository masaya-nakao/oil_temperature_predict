# %%
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# %%
# ett.csvファイルを読み込む
df = pd.read_csv('ett.csv')

# dateとOT列のみを抽出
df_filtered = df[['date', 'OT']]

# 新しいCSVファイルとして出力
df_filtered.to_csv('filtered_ett.csv', index=False)

# filtered_ett.csvファイルを読み込む
df_filtered = pd.read_csv('filtered_ett.csv')

# %%
# 各カラムの統計値を表示
print(df_filtered.describe())
# %%
# OTカラムに欠損値があるかどうかを確認
missing_values = df_filtered['OT'].isnull().sum()

if missing_values > 0:
    print(f"OTカラムには {missing_values} 個の欠損値があります。")
else:
    print("OTカラムには欠損値がありません。")

# %% 
# dateを横軸、OTを縦軸にプロット
plt.figure(figsize=(10, 6))
plt.plot(df_filtered['date'], df_filtered['OT'], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Oil Temperature (OT)')
plt.title('Oil Temperature Over Time')
plt.grid(True)
plt.show()

# %%
# IQRを計算
Q1 = df_filtered['OT'].quantile(0.25)
Q3 = df_filtered['OT'].quantile(0.75)
IQR = Q3 - Q1

# 異常値の閾値を設定
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 異常値を検出
outliers = df_filtered[(df_filtered['OT'] < lower_bound) | (df_filtered['OT'] > upper_bound)]

# 異常値を表示
print("異常値:")
print(outliers)

# %%
# filtered_ett.csvファイルを読み込む
df = pd.read_csv('filtered_ett.csv')

# dateカラムをdatetime型に変換
df['date'] = pd.to_datetime(df['date'])

# dateカラムをインデックスに設定
df.set_index('date', inplace=True)

# STL分解を実行
stl = STL(df['OT'], seasonal=23)  # 24時間の季節性を仮定
result = stl.fit()

# 分解結果をプロット
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
result.trend.plot(ax=ax1)
ax1.set_ylabel('Trend')
result.seasonal.plot(ax=ax2)
ax2.set_ylabel('Seasonal')
result.resid.plot(ax=ax3)
ax3.set_ylabel('Residual')
plt.xlabel('Date')
plt.show()
# %%
# 月ごとの平均温度を計算
monthly_avg = df.resample('M', on='date')['OT'].mean()

# 月ごとの平均温度をプロット
plt.figure(figsize=(12, 6))
monthly_avg.plot(kind='bar')
plt.xlabel('Month')
plt.ylabel('Average Oil Temperature (OT)')
plt.title('Average Oil Temperature by Month')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
# %% 初回実行
# dateカラムをdatetime型に変換
df['date'] = pd.to_datetime(df['date'])

# dateカラムをインデックスに設定
df.set_index('date', inplace=True)

# 訓練用データとテスト用データに分割（例: 最後の30日分をテスト用データとする）
train_data = df['OT'][:-30]
test_data = df['OT'][-30:]

# SARIMAモデルの定義とフィッティング
model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
model_fit = model.fit(disp=False)

# テストデータに対する予測
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

# 予測結果と実際の値をプロット
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data, label='Train')
plt.plot(test_data.index, test_data, label='Test')
plt.plot(test_data.index, predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Oil Temperature (OT)')
plt.title('SARIMA Model - Oil Temperature Prediction')
plt.legend()
plt.grid(True)
plt.show()

# 予測精度の評価（MAE）
mae = mean_absolute_error(test_data, predictions)
print(f'Mean Absolute Error: {mae}')
# %%
# dateカラムをインデックスに設定
df.set_index('date', inplace=True)

# 自己相関関数（ACF）をプロット
plt.figure(figsize=(12, 6))
plot_acf(df['OT'], lags=50, ax=plt.gca())
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF) of OT')
plt.grid(True)
plt.show()

# 偏自己相関関数（PACF）をプロット
plt.figure(figsize=(12, 6))
plot_pacf(df['OT'], lags=50, ax=plt.gca())
plt.xlabel('Lags')
plt.ylabel('Partial Autocorrelation')
plt.title('Partial Autocorrelation Function (PACF) of OT')
plt.grid(True)
plt.show()
# %%
# 訓練用データとテスト用データに分割（例: 最後の30日分をテスト用データとする）
train_data = df['OT'][:-30]
test_data = df['OT'][-30:]

# SARIMAモデルの定義とフィッティング
model = SARIMAX(train_data, order=(2, 1, 1), seasonal_order=(1, 1, 1, 24))
model_fit = model.fit(disp=False)

# テストデータに対する予測
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

# 予測結果と実際の値をプロット
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data, label='Train')
plt.plot(test_data.index, test_data, label='Test')
plt.plot(test_data.index, predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Oil Temperature (OT)')
plt.title('SARIMA Model - Oil Temperature Prediction')
plt.legend()
plt.grid(True)
plt.show()

# 予測精度の評価（MAE）
mae = mean_absolute_error(test_data, predictions)
print(f'Mean Absolute Error: {mae}')