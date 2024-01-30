import numpy as np


def pc_normalize(pc):
    """对点云归一化"""
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)  #求中心，对pc数组的每行求平均值，通过这条函数最后得到一个1×3的数组[x_mean,y_mean,z_mean];
    pc = pc - centroid  #点云平移  或  # 求得每一点到中点的绝对距离
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  # 将同一列的元素取平方相加得(x^2+y^2+z^2)，再开方，取最大，得最大标准差
    #pc ** 2 平移后的点云求平方   #np.sum(pc ** 2, axis=1)：每列求和
    pc = pc / m   # 归一化，这里使用的是Z-score标准化方法，即为(x-mean)/std
    return pc
