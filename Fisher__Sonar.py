#18050100226 ZihaoWu
#在Sonar数据上验证Fisher线性判别算法  Sonar数据2类，60维，208个样本————第一类97个样本，第二类111个样本
#训练和测试样本划分方法：留一法

import numpy as np
import pandas as pd

def Fisher(X1, X2, n, condition):
    #分别计算正在判别的两类的均值向量
    m1 = np.mean(X1, axis=0)
    m2 = np.mean(X2, axis=0)
    m1 = m1.reshape(n, 1)        #所给数据为行向量，这里转化为列向量便于计算
    m2 = m2.reshape(n, 1)

    #计算样本类内离散度矩阵Si和总样本类内离散度矩阵Sw
    S1 = np.zeros((n, n))         #初始化S1，S2
    S2 = np.zeros((n, n))
    if condition == 0:                 #第一类留1个样本test
        for i in range(0, 96):
            S1 += (X1[i].reshape(n, 1) - m1).dot((X1[i].reshape(n, 1) - m1).T)
        for i in range(0, 111):
            S2 += (X2[i].reshape(n, 1) - m2).dot((X2[i].reshape(n, 1) - m2).T)
    if condition == 1 :                 #第二类留1个样本test
        for i in range(0, 97):
            S1 += (X1[i].reshape(n, 1) - m1).dot((X1[i].reshape(n, 1) - m1).T)
        for i in range(0, 110):
            S2 += (X2[i].reshape(n, 1) - m2).dot((X2[i].reshape(n, 1) - m2).T)
    Sw = S1 + S2

    #计算最佳变换向量W
    W = np.linalg.inv(Sw).dot(m1 - m2)

    #计算阈值W0
    W0 = 0.5*(W.T.dot(m1) + W.T.dot(m2))

    return W, W0

#定义判别准则
def Classify(X, W):
    y = W.T.dot(X)
    return y

#导入数据集
sonar = pd.read_csv('sonar.all-data', header=None, encoding = 'gbk', sep = ',') #读取sonar数据
sonar1 = sonar.iloc[:208, :60]
sonar2 = np.mat(sonar1)

#将数据的矩阵形式切片，将两类样本分开
S1 = sonar2[0:97, 0:60]
S2 = sonar2[97:208, 0:60]

#存储投影点y
Y12_1 = np.zeros(97)
Y12_2 = np.zeros(111)

#留一法进行训练测试
right = 0
for i in range(208):
    if i <= 96:
        condition = 0
        test = S1[i].reshape(60, 1)
        train = np.delete(S1, i, axis=0)
        W, W0 = Fisher(train, S2, 60, condition)
        if (Classify(test, W)) >= W0:
            right += 1
        Y12_1[i] = Classify(test, W)
    else:
        condition = 1
        test = S2[i-97].reshape(60, 1)
        train = np.delete(S2, i-97, axis=0)
        W, W0 = Fisher(S1, train, 60, condition)
        if (Classify(test, W)) < W0:
            right += 1
        Y12_2[i-97] = Classify(test, W)
accuracy = right/208
print("Accuracy is：%.4f" % accuracy)

#画图
import matplotlib.pyplot as plt

y1 = np.zeros(97)
y2 = np.zeros(111)

plt.figure(1)
plt.ylim((-1, 1))            # y坐标的范围
plt.scatter(Y12_1, y1, c='r', alpha=1, marker='.')
plt.scatter(Y12_2, y2, c='b', alpha=1, marker='.')
plt.xlabel('Class:R&M  ' + 'Accuracy is ' + str(accuracy))
plt.legend(['R', 'M'])
plt.savefig('sonar.jpg', dpi=1500)
plt.show()






