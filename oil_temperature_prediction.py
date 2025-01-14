import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np

# ett.csvファイルを読み込む
df = pd.read_csv('ett.csv')

# dateカラムをdatetime型に変換
df['date'] = pd.to_datetime(df['date'])

# dateカラムをインデックスに設定
df.set_index('date', inplace=True)

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

# 予測精度の評価（MSE）
mse = mean_squared_error(test_data, predictions)
print(f'Mean Squared Error: {mse}')