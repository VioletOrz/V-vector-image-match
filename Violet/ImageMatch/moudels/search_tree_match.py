
from image_match.goldberg import ImageSignature

import os
import random
import time
import cv2
import base64
import io
import numpy as np
import sys


try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
    sys.path.append(parent_parent_dir)
except: pass

from Violet.Violet_base import list_all_files, write_pkl_file, read_pkl_file, get_parent_and_grandparent_dir
from Violet.ImageMatch.moudels.match import ImageMatcher
from Violet.ImageMatch.moudels.cosin import cosine_similarity

def base64_to_ndarray(base64_string):
    """
    将 Base64 编码的图像数据转换为 NumPy ndarray。

    Args:
        base64_string (str): Base64 编码的图像字符串。

    Returns:
        numpy.ndarray:  转换后的 NumPy ndarray， 如果转换失败， 则返回 None。
    """
    try:
      # 1. 解码 Base64 字符串
      image_bytes = base64.b64decode(base64_string)

      # 2. 使用 io.BytesIO 从 bytes 读取数据
      image_buffer = io.BytesIO(image_bytes)

      # 3.  读取图像数据，使用 OpenCV 读取图像数据， 并指定 cv2.IMREAD_UNCHANGED 读取透明通道
      img = cv2.imdecode(np.frombuffer(image_buffer.read(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

      return img # 返回图像

    except Exception as e:
      print(f"Error decoding base64: {e}")
      return None

def ndarray_to_base64(ndarray_data, image_format = ".png"):
    """
    将 NumPy ndarray 编码为 Base64 字符串。

    Args:
        ndarray_data (numpy.ndarray):  要编码的 NumPy ndarray。
        image_format (str, optional): 使用哪种图像格式编码, 默认为 ".png". 可以选择 ".jpg", ".bmp" 等等。

    Returns:
        str: Base64 编码的字符串， 如果编码失败，则返回 None。
    """
    try:
      # 1. 使用 opencv 将数据编码为 image 格式
      if len(ndarray_data.shape) == 2 or len(ndarray_data.shape) == 3:
        _, encoded_image = cv2.imencode(image_format, ndarray_data)
        image_bytes = encoded_image.tobytes() # 将图像编码为字节数据
      elif len(ndarray_data.shape) == 1:
         image_bytes = ndarray_data.tobytes()
      else:
         print("Error: Unknown ndarray format.")
         return None

      # 2. 编码为 Base64 字符串
      base64_string = base64.b64encode(image_bytes).decode('utf-8')

      return base64_string
    except Exception as e:
        print(f"Error encoding to base64: {e}")
        return None

class Map_matcher():

    def __init__(self):
        self.G_data = []
        self.map = [] #存储每个节点的父节点在fa_node中的索引
        self.fa_node = [] #存所有父节点
        #self.map = [] 
        self.search_tree = []
        self.min_distance = float('inf')
        self.insert_fa_node = []
        self.gis = ImageSignature()
        self.distance_function = self.gis.normalized_distance
        #self.gis.normalized_distance
    def load_pkl(self, pkl_path):
        self.G_data = read_pkl_file(pkl_path)
        #self.map = [0 for _ in range(len(self.G_data))]
        self.map = [-1 for _ in range(len(self.G_data))]
        
    def process_img_2_pkl(self, img_files_path, pkl_path):
        if type(img_files_path) == str:
            all_img_path = list_all_files(img_files_path)
        elif type(img_files_path) == list:
            all_img_path = img_files_path
        else:
            print('Error: img_files_path must be str or list')
            return
        gis = self.gis

        data = []
        for path in all_img_path:

            base_name, parent_name,_  = get_parent_and_grandparent_dir(path)
            
            print(base_name)
            tmp_sign = gis.generate_signature(path)
            id = parent_name + '/' + base_name[:-4]
            data.append({'id': id, 'vector': tmp_sign})
        write_pkl_file(pkl_path, data)
        self.G_data = data
        self.map = [-1 for _ in range(len(self.G_data))]

    def process_img_2_pkl_from_base64db(self, data_list, pkl_path):

        gis = self.gis
        G_data = []
        for i, data in enumerate(data_list):
            print(f'正在处理第{i+1}个图像')
            id = data['id']
            img = base64_to_ndarray(data['img'])
            #base_name, parent_name,_  = get_parent_and_grandparent_dir(img)
            
            #print(base_name)
            tmp_sign = gis.generate_signature(img)
            #id = parent_name + '/' + base_name[:-4]
            G_data.append({'id': id, 'vector': tmp_sign})
        write_pkl_file(pkl_path, G_data)
        self.G_data = G_data
        self.map = [-1 for _ in range(len(self.G_data))]
    def save_fa_map(self, pkl_path):
        data = {'fa':self.fa_node, 'map':self.map}
        write_pkl_file(pkl_path, data)

    def load_fa_map(self, pkl_path):
        data = read_pkl_file(pkl_path)
        self.fa_node = data['fa']
        self.map = data['map']

    def save_search_tree(self, pkl_path):
        data = {'tree': self.search_tree, 'min':self.min_distance}
        write_pkl_file(pkl_path, data)

    def load_search_tree(self, pkl_path):

        data = read_pkl_file(pkl_path)
        self.search_tree = data['tree']
        self.min_distance = data['min']
        
    def greedy_furthest_points(self, m, points = None, ):
        #001
        """
        使用贪心算法寻找彼此距离尽可能远的 m 个点。

        Args:
           points (list): 所有节点的坐标列表.
            m (int): 需要选择的节点数量。
            distance_function:  距离计算函数

        Returns:
            list: 选择的 m 个节点的索引列表。
        """
        distance_function = self.distance_function

        if points == None:
            points = self.G_data

        n = len(points)
        if m > n:
            print("Error: m must be less or equal than n")
            return None
        selected_indices = []
        # 1. 随机选择一个点
        first_index = random.randint(0, n - 1)
        selected_indices.append(first_index)

        # 2. 循环选择 m - 1 个点
        cnt = 1
        for _ in range(m - 1):
        #while self.min_distance > 0.4:
            print(f'正在寻找，第{cnt+1}个节点')
            cnt+=1
            max_min_distance = -1
            best_index = -1
            for i in range(n):
                if i in selected_indices:
                    continue
                min_distance = float('inf')
                for selected_index in selected_indices:
                   distance =  distance_function(points[i]['vector'], points[selected_index]['vector'])
                   min_distance = min(min_distance, distance) #当前节点到当前集合中头节点的最小距离
                if min_distance > max_min_distance: #如果当前节点到头节点之间的最小距离大于当前头节点到其他节点的距离，则更新头节点和头节点到其他节点的距离，筛选出和所有头节点都尽可能不相邻的点
                    max_min_distance = min_distance
                    best_index = i

            if self.min_distance > max_min_distance: #记录头节点之间最小距离
                self.min_distance = max_min_distance
                print(f'当前最小距离{self.min_distance}')

            selected_indices.append(best_index)
            self.map[best_index] = len(selected_indices) - 1
        
        self.fa_node = selected_indices
        return selected_indices
    
    def process_fa_node_map(self, ):
        #002
        
        distance_function = self.distance_function
        for data_id, data in enumerate(self.G_data):

            min_distance = float('inf')
            best_index = -1
            
            for fa_id, index in enumerate(self.fa_node):
                distance =  distance_function(data['vector'], self.G_data[index]['vector'])
                
                if distance < min_distance:
                    min_distance = distance
                    best_index = fa_id
            
            self.map[data_id] = best_index
            print(f'data_id:{data_id},父节点为{best_index}')
        return self.map
    
    def generate_search_tree(self):
        #003

        for index in range(len(self.fa_node)):
            tmp_tree = []
            tmp_tree.append(self.G_data[self.fa_node[index]])
            print(f'正在生成第{index+1}个节点的搜索树,当前父节点为{self.fa_node[index]}')
            for data_id, data in enumerate(self.G_data):
                #print(f'data_id:{data_id}')
                if self.map[data_id] == index and data_id != self.fa_node[index]:
                    print(f'加入到父节点:{self.fa_node[index]}')
                    tmp_tree.append(data)
            self.search_tree.append(tmp_tree)

        return self.search_tree
    

    def search_img(self, img_or_path, ):

        distance_function = self.distance_function
        min_fa_distance = float('inf')
        gis = self.gis
        
        img_sign = gis.generate_signature(img_or_path)
        tree_id = -1
        for index, tree in enumerate(self.search_tree):
            fa_node = tree[0]
            distance =  distance_function(img_sign, fa_node['vector'])
            if distance < min_fa_distance:
                
                min_fa_distance = distance
                tree_id = index

        if min_fa_distance > self.min_distance:
            return None, min_fa_distance

        min_distance = float('inf')
        img_id = None
        for index, data in enumerate(self.search_tree[tree_id]):
            distance =  distance_function(img_sign, data['vector'])
            if distance < min_distance:
                #print(distance)
                min_distance = distance
                img_id = data['id']

        return img_id, min_distance      
    def search_img_topk(self, img_or_path, thoshold = 0.55):

        #sorted(tuple_list, key=lambda x: x[1])
        
        topk_list = []

        distance_function = self.distance_function
        min_fa_distance = float('inf')
        gis = self.gis
        
        img_sign = gis.generate_signature(img_or_path)
        tree_id = -1
        for index, tree in enumerate(self.search_tree):
            fa_node = tree[0]
            distance =  distance_function(img_sign, fa_node['vector'])
            if distance < min_fa_distance:
                
                min_fa_distance = distance
                tree_id = index

        #min_distance = float('inf')
        #img_id = None
        for index, data in enumerate(self.search_tree[tree_id]):
            distance =  distance_function(img_sign, data['vector'])
            if distance <= thoshold:
                #print(distance)
                min_distance = distance
                #img_id = data['id']
                topk_list.append((data['id'], distance))
                
        if topk_list != []:
            return sorted(topk_list, key=lambda x: x[1])  
        else:
            return []
    
    def search_img_without_tree(self, img_or_path, ):
        distance_function = self.distance_function
        min_distance = float('inf')
        gis = ImageSignature()
        img_sign = gis.generate_signature(img_or_path)
        best_id = -1
        for data in self.G_data:
            distance =  distance_function(img_sign, data['vector'])
            if distance < min_distance:
                min_distance = distance
                best_id = data['id']

        return best_id
    
    def insert(self, img_or_path = None, signature = None, file_name = 'insert_img', ):
        distance_function = self.distance_function
        
        #if img_or_path == None and signature == None:
        #    print('Error: img_or_path and signature cannot be None at the same time.')
        #    return
        
        #if img_or_path != None:
        gis = ImageSignature()
        signature = gis.generate_signature(img_or_path)
        #else:
        #    signature = signature

        img_id = file_name + '/' + str(len(self.G_data)).zfill(6)

        self.G_data.append({'id': img_id, 'vector': signature})
        
        if self.insert_fa_node == []:
            self.fa_node.append(len(self.G_data) - 1)
            self.insert_fa_node.append(len(self.G_data) - 1)
            self.map.append(len(self.G_data) - 1)
            self.search_tree.append([self.G_data[-1]])
        else:
            min_distance = float('inf')
            insert_id = -1
            for data_id in self.insert_fa_node: #找和新插入头节点之间的最小距离
                distance =  distance_function(signature, self.G_data[data_id]['vector'])
                if distance < min_distance: 
                    min_distance = distance
                    insert_id = data_id
            
            if min_distance > self.min_distance: #如果最小距离大于原有头节点两两之间的最小距离，则作为新的头节点加入
                
                self.fa_node.append(len(self.G_data) - 1)
                self.insert_fa_node.append(len(self.G_data) - 1)
                self.map.append(len(self.G_data) - 1)
                self.search_tree.append([self.G_data[-1]])
            
            else: #否则则加入新增头节点的列表中
                id = self.fa_node.index(insert_id)
                self.search_tree[id].append(self.G_data[-1])

        return img_id + '.jpg'
    
    def insert_new(self, img_or_path = None, signature = None, img_id = 'insert_img', ):
        #不调用G_data和fa_node fa_map进行插入，直接加入搜索树中
        distance_function = self.distance_function
        
        #img = base64_to_ndarray(img_base64)
        signature = self.gis.generate_signature(img_or_path)


        self.G_data.append({'id': img_id, 'vector': signature})


        min_distance = float('inf')
        insert_id = -1
        for id, data_list in enumerate(self.search_tree):
            fa_id = data_list[0]['id']
            fa_vector = data_list[0]['vector']
            distance = distance_function(signature, fa_vector)
            if distance < min_distance:
                min_distance = distance
                insert_id = id

        if min_distance > self.min_distance:
            self.search_tree.append[[{'id': img_id, 'vector': signature}]]
        else:
            self.search_tree[insert_id].append({'id': img_id, 'vector': signature})

        return img_id

    def insert_base64(self, img_id, img_base64):
        distance_function = self.distance_function
        
        img = base64_to_ndarray(img_base64)
        signature = self.gis.generate_signature(img)


        self.G_data.append({'id': img_id, 'vector': signature})
        
        if self.insert_fa_node == []:
            self.fa_node.append(len(self.G_data) - 1)
            self.insert_fa_node.append(len(self.G_data) - 1)
            self.map.append(len(self.G_data) - 1)
            self.search_tree.append([self.G_data[-1]])
        else:
            min_distance = float('inf')
            insert_id = -1
            for data_id in self.insert_fa_node: #找和新插入头节点之间的最小距离
                distance = distance_function(signature, self.G_data[data_id]['vector'])
                if distance < min_distance: 
                    min_distance = distance
                    insert_id = data_id
            
            if min_distance > self.min_distance: #如果最小距离大于原有头节点两两之间的最小距离，则作为新的头节点加入
                
                self.fa_node.append(len(self.G_data) - 1)
                self.insert_fa_node.append(len(self.G_data) - 1)
                self.map.append(len(self.G_data) - 1)
                self.search_tree.append([self.G_data[-1]])
            
            else: #否则则加入新增头节点的列表中
                id = self.fa_node.index(insert_id)
                self.search_tree[id].append(self.G_data[-1])

        return img_id
    
    def insert_base64_new(self, img_id, img_base64):
        #不调用G_data和fa_node fa_map进行插入，直接加入搜索树中
        distance_function = self.distance_function
        
        img = base64_to_ndarray(img_base64)
        signature = self.gis.generate_signature(img)


        self.G_data.append({'id': img_id, 'vector': signature})


        min_distance = float('inf')
        insert_id = -1
        for id, data_list in enumerate(self.search_tree):
            fa_id = data_list[0]['id']
            fa_vector = data_list[0]['vector']
            distance = distance_function(signature, fa_vector)
            if distance < min_distance:
                min_distance = distance
                insert_id = id

        if min_distance > self.min_distance:
            self.search_tree.append[[{'id': img_id, 'vector': signature}]]
        else:
            self.search_tree[insert_id].append({'id': img_id, 'vector': signature})

        return img_id

        
        """if self.insert_fa_node == []:
            self.fa_node.append(len(self.G_data) - 1)
            self.insert_fa_node.append(len(self.G_data) - 1)
            self.map.append(len(self.G_data) - 1)
            self.search_tree.append([self.G_data[-1]])
        else:
            min_distance = float('inf')
            insert_id = -1
            for data_id in self.insert_fa_node: #找和新插入头节点之间的最小距离
                distance =  distance_function(signature, self.G_data[data_id]['vector'])
                if distance < min_distance: 
                    min_distance = distance
                    insert_id = data_id
            
            if min_distance > self.min_distance: #如果最小距离大于原有头节点两两之间的最小距离，则作为新的头节点加入
                
                self.fa_node.append(len(self.G_data) - 1)
                self.insert_fa_node.append(len(self.G_data) - 1)
                self.map.append(len(self.G_data) - 1)
                self.search_tree.append([self.G_data[-1]])
            
            else: #否则则加入新增头节点的列表中
                id = self.fa_node.index(insert_id)
                self.search_tree[id].append(self.G_data[-1])

        return img_id"""
    def get_insert_search_tree(pkl_path):

        data = mm.search_tree[-(len(mm.insert_fa_node))]
        write_pkl_file(pkl_path, data)

    def real_time_update(pkl_path):
        data = read_pkl_file(pkl_path)
        mm.search_tree.append(data)

    
if __name__ == '__main__':
    "E:/Genshin_frames_circle/01"
    "Genshin_01.pkl"
    save_name = 'test_genshin'
    #process_img_2_pkl()
    mm = Map_matcher()
    mm.process_img_2_pkl("E:/Genshin_Impact/01", save_name + '.pkl')
    mm.load_pkl(save_name + '.pkl')
    mm.greedy_furthest_points(30) #001
    mm.process_fa_node_map() #002
    mm.save_fa_map(save_name + '_fa_map.pkl')
    mm.load_fa_map(save_name + '_fa_map.pkl')
    print(mm.fa_node)
    mm.generate_search_tree() #003
    mm.save_search_tree(save_name + '_search_tree.pkl')
    mm.load_search_tree(save_name + '_search_tree.pkl')


    print(mm.search_img(r'E:\Genshin_impact_circle\03\122040.jpg'))
    print(mm.min_distance)
    print(mm.insert(r'E:\Genshin_impact_circle\03\122040.jpg'))
    print(mm.search_img(r'E:\Genshin_impact_circle\03\122040.jpg'))
    print(mm.insert(r'E:\Genshin_impact_circle\03\112680.jpg'))
    print(mm.search_img(r'E:\Genshin_impact_circle\03\112680.jpg'))
    
    len1 = len(mm.G_data)
    len2 = 0
    for id, i in enumerate(mm.search_tree):
        print(f'第{id + 1}个节点的长度为：{len(i)}')
        len2 += len(i)

    print(f'len1 = {len1}##len2 = {len2}')

    all_path = list_all_files('E:/Genshin_frames_circle/02')

    for p in all_path:
        test = mm.search_img(p)
        print(os.path.basename(p), test)

    """err = 0
    all_time = 0
    import time
    for p in all_path:
        start_time = time.time()
        r = mm.search_img_without_tree(p)
        end_time = time.time()
        print(f'{p}##{r}##{end_time - start_time}')
        all_time += end_time - start_time
        bn = os.path.basename(p)[:-4]
        if bn != r:
            print(bn, r)
            err+=1
    print(f'err = {err}##all_time = {all_time}')
    print(mm.search_img('E:/Genshin_frames_circle_nan/000000.jpg'))"""
    
