from Violet.Violet_base import list_all_files
from image_match.goldberg import ImageSignature
import time
from pymilvus import MilvusClient, DataType
from image_match.goldberg import ImageSignature
import os
from PIL import Image 
import numpy as np

CLUSTER_ENDPOINT = "https://in03-bd78ddc089214b3.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
TOKEN = "5d2edda38394a9fa6ed43dc4c80d4b885194463964868842f18a9c0f758c908ffdb39a93cb2faf28e178d03295145e4a1d196140"

# 1. Set up a Milvus client
client = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    token=TOKEN 
)

# 2. Create a collection in quick setup mode
"""client.create_collection(
    collection_name="Genshin",
    dimension=648
)"""

def add_data(all_img_path):

    data = []
    gis = ImageSignature()
    
    for path in all_img_path:
        tmp = gis.generate_signature(path)
        id = int(os.path.basename(path)[:-4])
        data.append({
            "id": id,
            "vector": list(tmp)
        })
    print('#########')
    res = client.insert(
        collection_name="Genshin",
        data=data
    )

def add_data_01(all_img_path):

    data = []
    gis = ImageSignature()
    
    for path in all_img_path:
        tmp = gis.generate_signature(path)
        id = int(os.path.basename(path)[:-4])
        data.append({
            "id": id,
            "vector_1": list(tmp),
            "vector_2": list(tmp),
            "vector_3": list(tmp)
        })
    print('#########')
    res = client.insert(
        collection_name="Genshin_01",
        data=data
    )

def compress_and_flatten_image(image_path):
    """
    将图像压缩到 12x12 尺寸的灰度图，并展开为一维的 Python 列表。

    :param image_path: 输入图像文件路径
    :return: 展开的图像一维列表
    """
    # 打开图像并转换为灰度图
    image = Image.open(image_path).convert("L")
    
    # 压缩图像为 12x12 尺寸
    image = image.resize((12, 12))
    
    # 将图像转换为列表形式
    flattened_image = list(image.getdata())
    
    return flattened_image

def add_data_test(all_img_path):

    data = []
    gis = ImageSignature()
    id = 0
    tmp = gis.generate_signature(all_img_path[0])
    while id < 1000000:
        tmp = tmp.astype(int)
        id = id + 1
        data.append({
            "id": id,
            "vector": 1,
        })
    print('#########')
    #res = client.insert(
    #    collection_name="Genshin_test",
    #    data=data
    #)
    from Violet.Violet_base import write_json_file

    write_json_file("test.json", data)

def add_data_02(all_img_path):

    data = []
    #gis = ImageSignature()
    
    for path in all_img_path:
        tmp = compress_and_flatten_image(path)
        id = int(os.path.basename(path)[:-4])
        data.append({
            "primary_key": id,
            "vector_1": tmp,
            "vector_2": tmp,
            "vector_3": tmp
        })
    print('#########')
    res = client.insert(
        collection_name="Genshin_02",
        data=data
    )

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

if __name__ == '__main__':

    all_img_path = list_all_files("E:/Genshin_frames_circle")
    gis = ImageSignature()
    #a = compress_and_flatten_image("E:/Genshin_frames_circle/002487.jpg")
    #b = compress_and_flatten_image("E:/Genshin_frames_circle/002176.jpg")
    a = gis.generate_signature("E:/Genshin_frames_circle/002487.jpg")
    b = gis.generate_signature("E:/Genshin_frames_circle//002486.jpg")


    start_time = time.time()
    for i in range(100000):
        #gis.normalized_distance(a,b)
        cosine_similarity(a[::4], b[::4])
    end_time = time.time()
    print(f"Search time: {end_time - start_time} seconds")

    #while True: pass
    
    
    #add_data_02(all_img_path)
    
    a = list(gis.generate_signature("E:/Genshin_frames_circle/002487.jpg"))#compress_and_flatten_image("E:/Genshin_frames_circle/002487.jpg")#gis.generate_signature("E:/Genshin_frames_circle/001540.jpg")
    #>0.44 <1170 >540
    ress = []
    start_time = time.time()
    res = client.search(
        collection_name="Genshin_01",     # target collection
        data=[a],
        anns_field="vector_2",
        #filter='vector_1',               # query vectors
        limit=20,                           # number of returned entities
    )
    end_time = time.time()
    print(f"Search time: {end_time - start_time} seconds")
    for r in res[0]:
        ress.append((r['id'], r['distance']))

    for rr in ress:
        print(rr)
    print(len(ress))
    #向量-地区-区域-具体位置