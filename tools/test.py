import numpy as np

def probabilistic_mask(v1, v2):
    # 初始化mask为0
    mask = np.zeros_like(v1, dtype=int)
    
    # 遍历每个元素，判断v2是否大于v1
    for i in range(len(v1)):
        if v2[i] > v1[i]:
            # 计算概率为 (v2 - v1) / v2
            prob = (v2[i] - v1[i]) / v2[i]
            # 随机生成mask值
            mask[i] = 1 if np.random.rand() < prob else 0
        else:
            mask[i] = 0  # 若v2 < v1 则 mask[i] 必定为0
    
    return mask

# 示例向量
v1 = np.array([0.2, 0.5, 0.7])
v2 = np.array([0.3, 0.4, 0.8])

# 计算mask
for i in range(100):
    mask = probabilistic_mask(v1, v2)
    print("Mask:", mask)

