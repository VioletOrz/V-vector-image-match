import numpy as np
def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度。

    Args:
        vec1 (list): 第一个向量 (Python list)。
        vec2 (list): 第二个向量 (Python list)。

    Returns:
        float: 两个向量之间的余弦相似度，如果向量长度不一致或者长度为 0， 则返回 None。
    """

    # 1. 将列表转换为 NumPy 数组
    #if vec1 == vec2: return 1
    vec1 = np.array(vec1, dtype=float) # 使用 float 类型避免整数运算误差
    vec2 = np.array(vec2, dtype=float)
 
    # 2. 检查向量长度是否一致
    if len(vec1) != len(vec2):
        print("Error: Vectors must have the same length.")
        return None

    # 3. 检查向量长度是否为 0
    if len(vec1) == 0:
        print("Error: Vectors cannot be empty.")
        return None


    # 4. 计算向量点积
    dot_product = np.dot(vec1, vec2)

    # 5. 计算向量的模 (L2 范数)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # 6. 计算余弦相似度
    if norm_vec1 == 0 or norm_vec2 == 0:
        print("Error: Cannot compute cosine similarity for zero vectors.")
        return None

    similarity = dot_product / (norm_vec1 * norm_vec2)

    return similarity