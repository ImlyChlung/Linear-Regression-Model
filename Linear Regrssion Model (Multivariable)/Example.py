import pandas as pd
from LinearRegression import LinearRegression as LR

# 讀取 CSV 文件
df = pd.read_csv('Student_Performance.csv')

# 將 "Yes" 和 "No" 轉換為數值
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# 分離特徵和標籤
features = df.drop(columns=['Performance Index']).values # 將 DataFrame 轉換為 NumPy 數組，除去標籤列，剩下的是特徵
labels = df['Performance Index'].values.reshape((df.shape[0],1))

# 創建線性回歸模型實例
Example = LR(features, labels)

# 訓練模型
Example.train(0.001, 3000)  # 設置學習率和迭代次數

# 可視化模型結果
Example.visualization()  # 顯示數據點、回歸線及損失函數的變化

# 測試模型性能
Example.test()  # 顯示模型在訓練和測試數據上的誤差標準差

# 預測新數據
Example.prediction()  # 用戶輸入新特徵值進行預測
