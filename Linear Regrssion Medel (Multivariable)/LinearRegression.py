import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class LinearRegression:

    def __init__(self, features, labels, test_size=0.2):
        """
        初始化線性回歸模型，添加一列全為1的數據作為截距項。
        輸入的參數應為：
        features (ndarray，N列矩陣): 特徵數據 (N維)
        labels (ndarray，一列(直行)矩陣): 標籤數據
        初始化步驟包括：
        1. 為特徵數據添加一列全為1的數據作為截距項。
        2. 標準化輸入的特徵數據。
        3. 將數據劃分為訓練集和測試集，默認測試集佔比為20%。
        4. 初始化模型參數 theta 為全零矩陣。
        5. 初始化 cost_history 列表，用於記錄每次迭代的損失值。
        """
        self.standardize_data(features, labels)
        ones = np.ones((features.shape[0], 1))
        self.features = np.hstack([ones, self.features])
        features_train, features_test, labels_train, labels_test = train_test_split(self.features, self.labels, test_size=test_size)
        self.features = features_train
        self.labels = labels_train
        self.features_test = features_test
        self.labels_test = labels_test
        self.theta = np.zeros((self.features.shape[1],1))
        self.cost_history = []

    def standardize_data(self, features, labels):
        """
        標準化輸入的特徵數據。
        """
        self.scaler_features = StandardScaler()
        self.scaler_labels = StandardScaler()
        self.features = self.scaler_features.fit_transform(features)
        self.labels = self.scaler_labels.fit_transform(labels)

    def train(self, alpha, num_of_iteration):
        """
        訓練線性回歸模型，使用梯度下降法更新參數theta。
        參數：
        alpha (float): 學習率
        num_of_iteration (int): 迭代次數
        """
        print(f"模型訓練前的參數為: theta = {self.theta}")
        print(f"模型訓練前的損失函數為: {self.cost_function()}")
        print(f"模型訓練前的損失函數導數為: {self.compute_gradient()}")
        for _ in range(num_of_iteration):
            self.gradient_step(alpha)
            self.cost_history.append(self.cost_function())
        print(f"模型訓練後的參數(對於標準化後的特徵數據，第一行為常數項)為: theta = {self.theta}")
        print(f"模型訓練後的損失函數為: {self.cost_function()}")
        print(f"模型訓練後的損失函數導數為(理論應逐漸趨向0): {self.compute_gradient()}")
        return self.theta

    def cost_function(self):
        """
        計算損失函數（均方誤差）。
        返回： cost (float): 損失函數值
        """
        delta = self.labels - np.dot(self.features, self.theta)
        cost = (1/self.features.shape[0])*np.dot(delta.T, delta)
        return cost[0][0]

    def compute_gradient(self):
        """
        計算損失函數的偏導數（梯度）。
        返回： gradient (一列(直行)矩陣): 梯度值
        """
        delta = self.labels - np.dot(self.features, self.theta)
        gradient = -(2/self.features.shape[0])*np.dot(self.features.T, delta)
        return gradient

    def gradient_step(self, alpha):
        """
        執行一步梯度下降法更新參數theta。
        參數： alpha (float): 學習率
        """
        gradient = self.compute_gradient()
        self.theta = self.theta - alpha * gradient

    def prediction(self):
        """
        輸入一個特徵值，使用訓練好的模型進行預測。
        """
        features = [float(x) for x in input("請輸入特徵值，用逗號分隔:").split(",")]
        features = np.array(features).reshape(1, -1)
        features = self.scaler_features.transform(features)
        ones = np.ones((features.shape[0], 1))
        features = np.hstack([ones, features])
        delta = self.scaler_labels.inverse_transform(self.labels) - self.scaler_labels.inverse_transform(np.dot(self.features, self.theta))
        scaler_prediction = np.dot(features, self.theta)
        prediction = self.scaler_labels.inverse_transform(scaler_prediction.reshape(-1, 1))
        print(f"模型預測值為: {prediction[0][0]}")
        print(f"誤差的平均值為(理論應逐漸趨向0): {np.mean(delta)}")
        print(f"誤差的標準差為: {np.std(delta)}")
        print(f"因此模型預測實際值有68.26%在範圍: {prediction[0][0]-np.std(delta)} - {prediction[0][0]+np.std(delta)}")
        print(f"95.44%在範圍: {prediction[0][0]-2*np.std(delta)} - {prediction[0][0]+2*np.std(delta)}")
        return self.theta

    def visualization(self):
        """
        可視化模型結果，顯示損失函數隨迭代次數的變化。
        """
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.plot(range(len(self.cost_history)), self.cost_history, color='green', label='損失函數')
        plt.xlabel('迭代次數')
        plt.ylabel('損失函數值')
        plt.title('損失函數的變化')
        plt.legend()
        plt.show()

    def test(self):
        """
        模擬模型遇到未知數據的表現，測試線性迴歸模型的性能。
        """
        delta = self.scaler_labels.inverse_transform(self.labels) - self.scaler_labels.inverse_transform(np.dot(self.features, self.theta))
        delta_test = self.scaler_labels.inverse_transform(self.labels_test) - self.scaler_labels.inverse_transform(np.dot(self.features_test, self.theta))
        print(f"模型給出的誤差的標準差為: {np.std(delta)}")
        print(f"測試數據給出的誤差的標準差為: {np.std(delta_test)}")
