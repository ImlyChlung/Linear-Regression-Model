import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class UnivariateLinearRegression:

    def __init__(self, features, labels, test_size=0.2):
        """
        初始化線性回歸模型，添加一列全為1的數據作為截距項。
        輸入的參數應為：
        features (ndarray，一列(直行)矩陣): 特徵數據
        labels (ndarray，一列(直行)矩陣): 標籤數據
        初始化步驟包括：
        1. 為特徵數據添加一列全為1的數據作為截距項。
        2. 將數據劃分為訓練集和測試集，默認測試集佔比為20%。
        3. 初始化模型參數 theta 為全零矩陣。
        4. 初始化 cost_history 列表，用於記錄每次迭代的損失值。
        """
        ones = np.ones((features.shape[0], 1))
        features = np.hstack([ones, features])
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_size)
        self.features = features_train
        self.labels = labels_train
        self.features_test = features_test
        self.labels_test = labels_test
        self.theta = np.zeros((2,1))
        self.cost_history = []

    def train(self, alpha, num_of_iteration):
        """
        訓練線性回歸模型，使用梯度下降法更新參數theta。
        參數：
        alpha (float): 學習率
        num_of_iteration (int): 迭代次數
        """
        print(f"模型訓練前的參數為: c = {self.theta[0]}, m = {self.theta[1]}")
        print(f"模型訓練前的損失函數為: {self.cost_function()}")
        print(f"模型訓練前的學習率乘以損失函數導數為: {alpha*self.compute_gradient()}")
        for _ in range(num_of_iteration):
            self.gradient_step(alpha)
            self.cost_history.append(self.cost_function())
        print(f"模型訓練後的參數為: c = {self.theta[0]}, m = {self.theta[1]}")
        print(f"模型訓練後的損失函數為: {self.cost_function()}")
        print(f"模型訓練後的損失學習率乘以函數導數為(理論應逐漸趨向0): {alpha*self.compute_gradient()}")
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
        features = float(input("請輸入特徵值:"))
        features = np.hstack([1, features])
        delta = self.labels - np.dot(self.features, self.theta)
        prediction = np.dot(features, self.theta)
        print(f"模型預測值為: {prediction[0]}")
        print(f"誤差的平均值為(理論應逐漸趨向0): {np.mean(delta)}")
        print(f"誤差的標準差為: {np.std(delta)}")
        print(f"因此模型預測實際值有68.26%在範圍: {prediction[0]-np.std(delta)} - {prediction[0]+np.std(delta)}")
        print(f"95.44%在範圍: {prediction[0]-2*np.std(delta)} - {prediction[0]+2*np.std(delta)}")

    def visualization(self):
        """
        可視化模型結果，顯示數據點和回歸線，以及損失函數隨迭代次數的變化。
        """
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.scatter(self.features[:, 1], self.labels, color='blue', label='訓練數據')
        plt.scatter(self.features_test[:, 1], self.labels_test, color='orange', label='測試數據')
        plt.plot(self.features[:, 1], np.dot(self.features, self.theta), color='red', label='迴歸線')
        plt.xlabel('特徵')
        plt.ylabel('標籤')
        plt.title('一元線性迴歸')
        plt.legend()
        plt.show()

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
        delta = self.labels - np.dot(self.features, self.theta)
        delta_test = self.labels_test - np.dot(self.features_test, self.theta)
        print(f"模型給出的誤差的標準差為: {np.std(delta)}")
        print(f"測試數據給出的誤差的標準差為: {np.std(delta_test)}")
