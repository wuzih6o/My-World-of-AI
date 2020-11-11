#18050100226 ZihaoWu
#在Iris数据上验证Fisher线性判别算法  Iris数据3类，4维，150个样本————每类50个样本
#训练和测试样本划分方法：留一法
import numpy as np
import pandas as pd

def Fisher(X1, X2, n):
    #分别计算正在判别的两类的均值向量
    m1 = np.mean(X1, axis=0)
    m2 = np.mean(X2, axis=0)
    m1 = m1.reshape(n, 1)        #所给数据为行向量，这里转化为列向量便于计算
    m2 = m2.reshape(n, 1)

    #计算样本类内离散度矩阵Si和总样本类内离散度矩阵Sw
    S1 = np.zeros((n, n))         #初始化S1，S2
    S2 = np.zeros((n, n))
    for i in range(0, X1.shape[0]):
        S1 += (X1[i].reshape(n, 1) - m1).dot((X1[i].reshape(n, 1) - m1).T)
    for i in range(0, X2.shape[0]):
        S2 += (X2[i].reshape(n, 1) - m2).dot((X2[i].reshape(n, 1) - m2).T)
    Sw = S1 + S2

    #计算最佳变换向量W
    W = np.linalg.inv(Sw).dot(m1 - m2)

    #计算阈值W0
    W0 = 0.5*(W.T.dot(m1) + W.T.dot(m2))

    return W, W0

#定义判别准则
def Classify(X, W, W0):
    y = W.T.dot(X) - W0
    return y

#导入数据集
iris = pd.read_csv('iris.data', header = None, encoding = 'gbk', sep = ',') #读取iris数据
iris1 = iris.iloc[:150, :4]  #读取前150行，前4列数据
iris2 = np.mat(iris1)         #变为数据的矩阵形式


#将数据的矩阵形式切片，每一类分开
I1 = iris2[0:50, 0:4]
I2 = iris2[50:100, 0:4]
I3 = iris2[100:150, 0:4]

#存储投影点
Class1 = []
Class2 = []
Class3 = []
#测试
count1 = 0
count2 = 0
count3 = 0
for i in range(150):
    if i < 50:
        test = I1[i].reshape(4, 1)
        train1 = np.delete(I1, i, axis=0)
        train2 = I2
        train3 = I3
    elif i < 100:
        test = I2[i-50].reshape(4, 1)
        train1 = I1
        train2 = np.delete(I2, i-50, axis=0)
        train3 = I3
    elif i < 150:
        test = I3[i-100].reshape(4, 1)
        train1 = I1
        train2 = I2
        train3 = np.delete(I3, i-100, axis=0)
    W_12, W0_12 = Fisher(train1, train2, 4)
    W_13, W0_13 = Fisher(train1, train3, 4)
    W_23, W0_23 = Fisher(train2, train3, 4)
    Y12 = Classify(test, W_12, W0_12)
    Y13 = Classify(test, W_13, W0_13)
    Y23 = Classify(test, W_23, W0_23)
    if Y12 >= 0 and Y13 >= 0:
        Class1.append(Y12)
        if i < 50:
            count1 += 1
    if Y12 < 0 and Y23 >= 0:
        Class2.append(Y12)
        if i >= 50 and  i < 100:
            count2 += 1
    if Y13 < 0 and Y23 < 0:
        Class3.append(Y13)
        if i >= 100 and i < 150:
            count3 += 1

accuarcy1 = count1/50
accuracy2 = count2/50
accuracy3 = count3/50

print("第一类的分类准确率为：%.4f\n" %accuarcy1)
print("第二类的分类准确率为：%.4f\n" %accuracy2)
print("第三类的分类准确率为：%.4f" %accuracy3)

#画图
#Class1、2、3分别用红色、绿色、蓝色的点表示
import matplotlib.pyplot as plt

y1 = np.zeros(len(Class1))
y2 = np.zeros(len(Class2))
y3 = np.zeros(len(Class3))

plt.figure(1)
plt.ylim((-1, 1))            # y坐标的范围
plt.scatter(Class1, y1, c='r', alpha=1, marker='.')
plt.scatter(Class2, y2, c='g', alpha=1, marker='.')
plt.scatter(Class3, y3, c='b', alpha=1, marker='.')
plt.xlabel('Class:1&2&3')
plt.legend(['Class1', 'Class2', 'Class3'])
plt.savefig('iris.jpg', dpi=1500)
plt.show()
















