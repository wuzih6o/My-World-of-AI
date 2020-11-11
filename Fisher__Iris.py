#18050100226 ZihaoWu
#在Iris数据上验证Fisher线性判别算法  Iris数据3类，4维，150个样本————每类50个样本
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
        for i in range(0, 49):
            S1 += (X1[i].reshape(n, 1) - m1).dot((X1[i].reshape(n, 1) - m1).T)
        for i in range(0, 50):
            S2 += (X2[i].reshape(n, 1) - m2).dot((X2[i].reshape(n, 1) - m2).T)
    if condition == 1 :                 #第二类留1个样本test
        for i in range(0, 50):
            S1 += (X1[i].reshape(n, 1) - m1).dot((X1[i].reshape(n, 1) - m1).T)
        for i in range(0, 49):
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
iris = pd.read_csv('iris.data', header = None, encoding = 'gbk', sep = ',') #读取iris数据
iris1 = iris.iloc[:150, :4]  #读取前150行，前4列数据
iris2 = np.mat(iris1)         #变为数据的矩阵形式


#将数据的矩阵形式切片，每一类分开
I1 = iris2[0:50, 0:4]
I2 = iris2[50:100, 0:4]
I3 = iris2[100:150, 0:4]

#存储投影点y
Y12_1 = np.zeros(50)
Y12_2 = np.zeros(50)
Y13_1 = np.zeros(50)
Y13_2 = np.zeros(50)
Y23_1 = np.zeros(50)
Y23_2 = np.zeros(50)

#对第一类和第二类进行Fisher判别分析，使用留一法
right = 0
for i in range(100):
    if i <= 49:
        condition = 0     #情况1：第一类中选择一个做测试
        test = I1[i].reshape(4, 1)
        train = np.delete(I1, i, axis=0)
        W, W0 = Fisher(train, I2, 4, condition)
        if (Classify(test, W)) >= W0:       #判断测试结果是否属于第一类
            right += 1
        Y12_1[i] = Classify(test, W)    #计算出投影点
    else:
        condition = 1     #情况2：第二类中选择一个做测试
        test = I2[i-50].reshape(4, 1)
        train = np.delete(I2, i-50, axis=0)
        W, W0 = Fisher(I1, train, 4, condition)
        if (Classify(test, W)) < W0:       #判断测试结果是否属于第二类
            right += 1
        Y12_2[i-50] = Classify(test, W)

accuracy12 = right/100

print("Accuracy between 1&2 is：%.4f" % accuracy12)

#对第一类、第三类进行判别
right = 0
for i in range(100):
    if i <= 49:
        condition = 0
        test = I1[i].reshape(4, 1)
        train = np.delete(I1, i, axis=0)
        W, W0 = Fisher(train, I3, 4, condition)
        if (Classify(test, W)) >= W0:
            right += 1
        Y13_1[i] = Classify(test, W)
    else:
        condition = 1
        test = I3[i-50].reshape(4, 1)
        train = np.delete(I3, i-50, axis=0)
        W, W0 = Fisher(I1, train, 4, condition)
        if (Classify(test, W)) < W0:
            right += 1
        Y13_2[i-50] = Classify(test, W)
accuracy13 = right/100
print("Accuracy between 1&3 is: %.4f"%accuracy13)

#对第二类、第三类进行判别
right = 0
for i in range(100):
    if i <= 49:
        condition = 0
        test = I2[i].reshape(4, 1)
        train = np.delete(I2, i, axis=0)
        W, W0 = Fisher(train, I3, 4, condition)
        if (Classify(test, W)) >= W0:
            right += 1
        Y23_1[i] = Classify(test, W)
    else:
        condition = 1
        test = I3[i-50].reshape(4, 1)
        train = np.delete(I3, i-50, axis=0)
        W, W0 = Fisher(I2, train, 4, condition)
        if (Classify(test, W)) < W0:
            right += 1
        Y23_2[i-50] = Classify(test, W)
accuracy23 = right/100
print("Accuracy between 2&3 is: %.4f"%accuracy23)

#画图
#Class1、2、3分别用红色、绿色、蓝色的点表示
import matplotlib.pyplot as plt

y1 = np.zeros(50)
y2 = np.zeros(50)

plt.figure(1)
plt.ylim((-1, 1))            # y坐标的范围
plt.scatter(Y12_1, y1, c='r', alpha=1, marker='.')
plt.scatter(Y12_2, y2, c='g', alpha=1, marker='.')
plt.xlabel('Class:1&2  ' + 'Accuracy is ' + str(accuracy12))
plt.legend(['Class1', 'Class2'])
plt.savefig('iris1&2', dpi=1500)

plt.figure(2)
plt.ylim((-1, 1))
plt.scatter(Y13_1, y1, c='r', alpha=1, marker='.')
plt.scatter(Y13_2, y2, c='b', alpha=1, marker='.')
plt.xlabel('Class:1&3  ' + 'Accuracy is ' + str(accuracy13))
plt.legend(['Class1', 'Class3'])
plt.savefig('iris1&3', dpi=1500)

plt.figure(3)
plt.ylim((-1, 1))
plt.scatter(Y23_1, y1, c='g', alpha=1, marker='.')
plt.scatter(Y23_2, y2, c='b', alpha=1, marker='.')
plt.xlabel('Class:2&3  ' + 'Accuracy is ' + str(accuracy23))
plt.legend(['Class2', 'Class3'])
plt.savefig('iris2&3', dpi=1500)

plt.show()
















