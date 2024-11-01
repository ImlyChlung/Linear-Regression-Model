import pandas as pd
from UnivariateLinearRegression import UnivariateLinearRegression as ULR

# 讀取數據，這是一份學生溫習時間和成績的數據
data = pd.read_csv('data.csv', header=None)

# 分開特徵和標籤
features = data.iloc[:, 0].values.reshape(-1, 1)  # 將第一列作為特徵，重塑為一列矩陣
labels = data.iloc[:, 1].values.reshape(-1, 1)    # 將第二列作為標籤，重塑為一列矩陣

# 創建線性回歸模型實例
Example = ULR(features, labels)

# 訓練模型
Example.train(alpha=0.00001, num_of_iteration=100)  # 設置學習率和迭代次數

# 可視化數據和回歸線
Example.visualization()

# 預測新數據
Example.prediction()  # 用戶輸入新特徵值進行預測
