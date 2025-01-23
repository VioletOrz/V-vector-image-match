from image_match.goldberg import ImageSignature
import os
from pathlib import Path
import numpy as np
import sys
import cv2
import time
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
    sys.path.append(parent_parent_dir)
except: pass

from Violet_base import read_pkl_file,write_pkl_file,write_json_file,read_json_file
from ImageMatch.moudels.SSIM import calculate_ssim, process_in_threads



class ImageMatcher:
    def __init__(self, ):
        self.gis = ImageSignature()
        self.image_signatures = None
        self.story_blocks_image_signatures = None
        
    def __init_block(self, block_cnt):
        block_queue = queue.Queue()
        block_id_cnt = []
        for i in range(block_cnt):
            block_id_cnt.append(0)
        self.block_match_cnt = {'last_in': None, #上次进入block的位置
                                'last_out': None, #没用
                                'now':0, #当前位置/总匹配次数
                                'queue':block_queue,#实现检测窗口的队列
                                'block_id_cnt':block_id_cnt,#存储每个block的匹配次数，达到指定次数进入block
                                'block_id': None, #当前block的id,仅在进入block时生效
                                'block_None': 0 #在block中漏检的次数，连续漏检达到一定次数提前跳出block
                                }
        
    def __reset_block_queue(self,):
        self.block_match_cnt['queue'] =queue.Queue()
        block_id_cnt = []
        for i in range(len(self.block_match_cnt['block_id_cnt'])):
            block_id_cnt.append(0)
        self.block_match_cnt['block_id_cnt'] = block_id_cnt

    def load_image_signatures(self, path):
        try:
            self.image_signatures = read_pkl_file(path)
        except Exception as e:
            print("Error loading image signatures")
            print(e)
     
    def load_story_blocks_image_signatures(self, pkl_path, json_path):
        try:
            self.image_signatures = read_pkl_file(pkl_path)
            self.story_blocks_image_signatures = read_json_file(json_path)
            self.__init_block(len(self.story_blocks_image_signatures['story_block'])) 

        except Exception as e:
            print("Error loading story blocks image signatures")
            print(e)
            
    def process_image_to_story_block_signatures(self, image_path_list, use_ssim = False):
        def get_parent_and_grandparent_dir(file_path):
            # 获取父文件夹路径
            parent_dir = os.path.dirname(file_path)
            # 获取父文件夹名称
            parent_name = os.path.basename(parent_dir)
            # 获取祖父文件夹路径
            grandparent_dir = os.path.dirname(parent_dir)
            # 获取祖父文件夹名称
            grandparent_name = os.path.basename(grandparent_dir)
            return parent_name, grandparent_name

        section_path_list_dict = {} #章节名称和对应的所有图片路径列表
        for path in image_path_list:
            parent_name, grandparent_name = get_parent_and_grandparent_dir(path)
            if grandparent_name not in section_path_list_dict: section_path_list_dict[grandparent_name] = []
            section_path_list_dict[grandparent_name].append(path)

        #每个章节的图像按照时间排序
        all_sorted_image_path_list = []#从字典中按顺序存储排好序的图像路径
        for key, value in section_path_list_dict.items():
            value = sorted(value, key=lambda path: os.path.basename(path))
            for path in value:
                all_sorted_image_path_list.append(path)
        #跳转到末尾使用该路径读取图像

        all_sorted_relative_path_list = []
        for path in all_sorted_image_path_list:
            parent_name, grandparent_name = get_parent_and_grandparent_dir(path)
            all_sorted_relative_path_list.append(f'{grandparent_name}/{parent_name}/{os.path.basename(path)}')

        path_index_dict = {} #相对路径和id直接的映射
        index_path_dict = {}
        for index, path in enumerate(all_sorted_relative_path_list):
            path_index_dict[path] = str(index).zfill(6)
            index_path_dict[str(index).zfill(6)] = path
        
        index_dict = {} #分离剧情帧和非剧情帧的6位索引
        for index, path in enumerate(all_sorted_image_path_list):
            parent_name, grandparent_name = get_parent_and_grandparent_dir(path)
            if parent_name not in index_dict: index_dict[parent_name] = []
            index_dict[parent_name].append(str(index).zfill(6))
        
        story_block = [[],]
        story_image_length = len(index_dict['story'])
        block_length = int(story_image_length**0.5) if story_image_length**0.5 >= 1000 else 1000
        id_block_dict = {} #反向查询优化,通过图像的6位编码查询位于哪个block

        story_block[0] = index_dict['other']
        for index in index_dict['other']:
            id_block_dict[index] = 0

        for id in range(0, story_image_length, block_length):
            if story_image_length - id < 0.5 * block_length: break
            if story_image_length - id < 1.5 * block_length:
                tmp_list = index_dict['story'][id:]
                #other = index_dict['other'][id:]
            else: 
                tmp_list = index_dict['story'][id:id+block_length] #分块搜索优化,通过block编号索引到一个区块的内容
                #other = index_dict['other'][id:id+block_length]
            block_id = len(story_block)
            
            for index in tmp_list:
                id_block_dict[index] = block_id
            #for index in other: 
                #id_block_dict[index] = 0
            
            #story_block[0] += other
            story_block.append(tmp_list)

        self.story_blocks_image_signatures = {'story_block': story_block, #block的图像id，0是other 从1开始是剧情帧的id
                                              'id_block_dict': id_block_dict, #6位编码到block编号的映射，定位当前帧在哪个block
                                              'block_length': block_length, #单个block的长度，长度最小为1000，最后一个block内容最少为0.5block_length,最多为1.5block_length
                                              'index_path_dict': index_path_dict, #6位编码到相对路径的映射，用于定位到本地图片
                                              'path_index_dict': path_index_dict, #相对路径到6位编码的映射，用于查询图像的编码
                                              'all_sorted_relative_path_list': all_sorted_relative_path_list, #按视频时间顺序排序好的帧的相对路径
                                              'use_ssim': use_ssim, 
                                              'story_image_length': story_image_length, #剧情帧的数量
                                              'index_dict': index_dict}#分开存储了所有剧情帧和非剧情帧
        self.__init_block(len(self.story_blocks_image_signatures['story_block']))     
        self.process_image_to_signatures(all_sorted_image_path_list, add=False, use_ssim=use_ssim)

    def process_image_to_signatures(self, image_or_path, add = False, use_ssim = False):
        #把图像处理为sign
        #输入必须是 string地址/np.array类型的图片数据 或者 是一个list，list中的元素可以是string地址/np.array类型
        #add: 是否添加到已有的image_signatures中，默认为False
        #start_index: 添加到已有的image_signatures中的起始位置，默认为000000,这个参数没有进行异常检测，不建议使用
        #保存的数据类型是字典，index: 图片的签名
        #每个图像有一个单独的6位index，可以与其它数据项匹配

        if add == True:
            if self.image_signatures == None:
                raise Exception("Image signatures not loaded")
            new_image_signatures = self.image_signatures
            start_index = len(self.image_signatures)
        else:
            new_image_signatures = {}

        
        if type(image_or_path) in [str, np.ndarray]:
            image_signatures = self.gis.generate_signature(image_or_path)
            if use_ssim:
                img = self.cvt_2_mosaic_gary(image_or_path)
                new_image_signatures[str(start_index).zfill(6)] = {'img_sign':image_signatures,'gary':img}
            else:
                new_image_signatures[str(start_index).zfill(6)] = {'img_sign':image_signatures}
            self.image_signatures = new_image_signatures
        elif type(image_or_path) == list:
            for i in range(0, len(image_or_path)):
                if type(image_or_path[i] == str): print(f"process: {image_or_path[i]}")

                if type(image_or_path[i]) in [str, np.ndarray]:
                    image_signatures = self.gis.generate_signature(image_or_path[i])
                    if use_ssim: 
                        img = self.cvt_2_mosaic_gary(image_or_path[i])
                        new_image_signatures[str(i).zfill(6)] = {'img_sign':image_signatures,'gary':img}
                    else:
                        new_image_signatures[str(i).zfill(6)] = {'img_sign':image_signatures}
            self.image_signatures = new_image_signatures
        else: raise Exception("Invalid input type, type should be string/np.array image or list of string/np.array image")
        

    def save_image_signatures(self, path):
        #保存数据到本地
        try:
            write_pkl_file(path, self.image_signatures)
        except Exception as e:
            print("Error saving image signatures")
            print(e)

    def save_story_blocks_image_signatures(self, pkl_path, json_path):
        #保存数据到本地
        try:
            write_pkl_file(pkl_path, self.image_signatures)
            write_json_file(json_path, self.story_blocks_image_signatures)
        except Exception as e:
            print("Error saving story block image signatures")
            print(e)

    def block_match_image(self,  image_or_path, match_threshold=0.4, first_match=False, use_ssim=False, ssim_threshold=0.8, mix_rate=0.6, block_threshold=(10,15,300)):
        """
        根据给定的图像路径，使用故事块图像签名来匹配图像，以确定是否进入或退出一个故事块。
        
        参数:
        -  image_or_path: 图像的路径。
        - match_threshold: 匹配阈值，用于确定图像是否匹配。
        - batch_match: 是否使用批处理匹配。
        - first_match: 是否仅在第一批中查找匹配项。
        - use_ssim: 是否使用结构相似性指数（SSIM）进行匹配。
        - ssim_threshold: SSIM阈值，用于确定图像是否相似。
        - mix_rate: 混合率，用于结合匹配分数和SSIM分数。
        - block_threshold: 包含三个阈值的元组，用于控制进入和退出故事块的逻辑。
        
        返回:
        - 匹配结果，具体取决于是否使用SSIM和其他参数。
        """
        
        # 检查故事块图像签名和图像签名是否已加载
        if self.story_blocks_image_signatures is None or self.image_signatures is None:
            raise Exception("Story block image signatures not loaded")
        try:
            # 增加匹配计数
            self.block_match_cnt['now'] += 1

            # 解析block阈值参数
            block_detect_threshold = block_threshold[0]  # 进入block至少需要匹配到多少图片
            block_detect_limit = block_threshold[1]  # 检测窗口大小
            block_detect_time = block_threshold[2]  # 进入一次block持续的时间

            # 如果当前没有处于任何block中
            if self.block_match_cnt['block_id'] is None:
                # 准备other和story的图像签名
                other = self.story_blocks_image_signatures['story_block'][0]
                story = self.story_blocks_image_signatures['story_block']

                other_image_signatures = {}
                for i, id in enumerate(other):
                    if i % 4 != 0: continue
                    other_image_signatures[id] = self.image_signatures[id]

                full_story_image_signatures = {}
                for block in story[1:]:
                    for i, id in enumerate(block):
                        #if i % 2 != 0: continue
                        full_story_image_signatures[id] = self.image_signatures[id]

                fast_use_ssim = False
                # 使用快速匹配进行other和story的匹配
                other_fast_match = self.match_image(image_or_path= image_or_path,
                                                    match_threshold=match_threshold,
                                                    batch_match=False,
                                                    first_match=first_match,
                                                    use_ssim=fast_use_ssim,
                                                    ssim_threshold=ssim_threshold,
                                                    match_image_signatures=other_image_signatures,
                                                    )
                # other_fast_match = (None, 0.6, None)
                full_story_fast_match = self.match_image(image_or_path= image_or_path,
                                                        match_threshold=match_threshold,
                                                        batch_match=False,
                                                        first_match=first_match,
                                                        use_ssim=fast_use_ssim,
                                                        ssim_threshold=ssim_threshold,
                                                        match_image_signatures=full_story_image_signatures,
                                                        )

                # 根据匹配结果计算分数
                if use_ssim:
                    if other_fast_match is not None:
                        if fast_use_ssim and other_fast_match[2] is not None:
                            score_other = (1 - other_fast_match[1]) * mix_rate + other_fast_match[2] * (1 - mix_rate)
                        else:
                            score_other = (1 - other_fast_match[1]) * mix_rate
                    else:
                        score_other = 0

                    if full_story_fast_match is not None:
                        if fast_use_ssim and full_story_fast_match[2] is not None:
                            score_story = (1 - full_story_fast_match[1]) * mix_rate + full_story_fast_match[2] * (1 - mix_rate)
                        else:
                            score_story = (1 - full_story_fast_match[1]) * mix_rate
                    else:
                        score_story = 0
                else:
                    if other_fast_match is not None:
                        score_other = other_fast_match[1]
                    else:
                        score_other = 0
                    if full_story_fast_match is not None:
                        score_story = full_story_fast_match[2]
                    else:
                        score_story = 0

                # 根据分数判断是否进入block
                if score_story > score_other:
                    # 剧情帧
                    img_index = full_story_fast_match[0]
                    block_id = self.story_blocks_image_signatures['id_block_dict'][img_index]
                    self.block_match_cnt['queue'].put(block_id)
                    self.block_match_cnt['block_id_cnt'][block_id] += 1

                    if self.block_match_cnt['block_id_cnt'][block_id] >= block_detect_threshold:  # 判断是否进入block
                        self.block_match_cnt['block_id'] = block_id
                        self.block_match_cnt['last_in'] = self.block_match_cnt['now']
                        print('#' * 100)
                        print(f'进入剧情块：{block_id}')

                    if self.block_match_cnt['queue'].qsize() > block_detect_limit:  # 超出窗口大小则清除队列头部
                        moved_id = self.block_match_cnt['queue'].get()
                        self.block_match_cnt['block_id_cnt'][moved_id] -= 1

                    return full_story_fast_match  # 返回剧情帧匹配结果
                else:
                    img_index = other_fast_match[0]
                    self.block_match_cnt['block_id_cnt'][0] += 1

                    if self.block_match_cnt['queue'].qsize() > block_detect_limit:  # 超出窗口大小则清除队列头部
                        moved_id = self.block_match_cnt['queue'].get()
                        self.block_match_cnt['block_id_cnt'][moved_id] -= 1

                    return None  # 不输出非剧情帧
            else:
                # 如果当前已经处于某个block中
                # if self.block_match_cnt['now'] == (self.block_match_cnt['last_in'] + 1):  # 节约时间
                block_story = self.story_blocks_image_signatures['story_block'][self.block_match_cnt['block_id']]
                block_image_signatures = {}
                for id in block_story:
                    block_image_signatures[id] = self.image_signatures[id]

                block_story_match = self.match_image(image_or_path= image_or_path,
                                                    match_threshold=match_threshold * 0.7,
                                                    batch_match=False,
                                                    first_match=first_match,
                                                    use_ssim=use_ssim,
                                                    ssim_threshold=ssim_threshold,
                                                    match_image_signatures=block_image_signatures,
                                                    )

                if self.block_match_cnt['now'] - self.block_match_cnt['last_in'] >= block_detect_time:  # 超出时间限制 退出block
                    block_id = self.block_match_cnt['block_id']
                    print('#' * 100)
                    print(f'退出剧情块：{block_id}，原因：超时')
                    self.block_match_cnt['block_id'] = None
                    # self.block_match_cnt['last_out'] = self.block_match_cnt['now']
                    self.__reset_block_queue()
                    self.block_match_cnt['block_None'] = 0

                if block_story_match is None:
                    self.block_match_cnt['block_None'] += 1
                    if self.block_match_cnt['block_None'] >= block_detect_limit // 3:  # 次数过多提前跳出block
                        block_id = self.block_match_cnt['block_id']
                        print('#' * 100)
                        print(f'退出剧情块：{block_id}，原因：漏检')
                        self.block_match_cnt['block_id'] = None
                        # self.block_match_cnt['last_out'] = self.block_match_cnt['now']
                        self.__reset_block_queue()
                        self.block_match_cnt['block_None'] = 0

                    return None
                else:
                    if self.block_match_cnt['block_None'] >= 1:  # 记录连续漏检次数 次数过多提前跳出block
                        self.block_match_cnt['block_None'] -= 1

                    return block_story_match

        except Exception as e:
            print("Error block matching image")
            print(e)

        
    def match_image(self, image_or_path, match_threshold = 0.4, batch_match = False, first_match = False, use_ssim = False, ssim_threshold = 0.8, match_image_signatures = None,):
        
        #zero = [np.int8(0)] * 648
        #匹配图像，一般来说 <0.4 为非常接近
        #输入的必须是 np.ndarray类型的图片数据 或者 图像的string地址
        #返回值是一个元组，第一个元素是匹配到的图片的index，第二个元素是匹配的image match相似度, 第三个（启用use_ssim时才会有第三个）是ssim的匹配度
        #batch_match: 是否批量匹配，默认为False
        #first_match: 是否只匹配第一个，默认为False
        #启用batch_match参数时，first_match参数失效
        #use_ssim: 是否启用ssim算法，默认为False,启用时first_match参数失效，建议把match阈值提高到0.55/0.6
        #ssim_threshold: ssim算法的阈值，默认为0.8,建议使用0.8-0.9
        if match_image_signatures:
            backup = self.image_signatures #备份
            self.image_signatures = match_image_signatures
        if self.image_signatures == None:
            raise Exception("Image signatures not loaded")
        
        try:
            img_sign = self.gis.generate_signature(image_or_path)
            if batch_match == False:
                if use_ssim:
                    min_t = 1
                    min_id = None
                    batch_match_list = []
                    for id, value in self.image_signatures.items():
                        k = id
                        v = value['img_sign']  
                        if id == '000139':
                            pass
                        if np.array_equal(img_sign, v):
                            t = 0.0
                        else:
                            t = self.gis.normalized_distance(img_sign, v)
                        if t < match_threshold:
                            batch_match_list.append((k,t))
                            if t < min_t: 
                                min_t = t
                                min_id = k
                    
                    img = self.cvt_2_mosaic_gary(image_or_path)
                    print(len(batch_match_list))


                    match_id = None
                    maxssim = 0
                    t = None
                    for i in batch_match_list:
                        id = i[0]
                            
                        score = calculate_ssim(img, self.image_signatures[id]['gary'])
                        if score > maxssim and score > ssim_threshold:
                            maxssim = score
                            match_id = id
                            t = i[1]
                    
                    if match_id == None: 

                        if min_id == None:
                            raise Exception("No match found")
                        else: 
                            print('SSIM未匹配到图像')
                            if match_image_signatures: self.image_signatures = backup
                            return min_id, min_t, None
                    else:
                        if match_image_signatures: self.image_signatures = backup 
                        return match_id, t, maxssim
                else:
                    if first_match == True:
                        for id, value in self.image_signatures.items():
                            k = id
                            v = value['img_sign']
                            if np.array_equal(img_sign, v):
                                t = 0.0
                            else:
                                t = self.gis.normalized_distance(img_sign, v)
                            if t < match_threshold:
                                if match_image_signatures: self.image_signatures = backup
                                return k, t
                        raise Exception("No match found")
                    else:
                        indx = None
                        for id, value in self.image_signatures.items():
                            k = id
                            v = value['img_sign']
                            if np.array_equal(img_sign, v):
                                t = 0.0
                            else:
                                t = self.gis.normalized_distance(img_sign, v)
                            if t < match_threshold:
                                indx = k
                                match_threshold = t
                        if indx == None: raise Exception("No match found")
                        else: 
                            if match_image_signatures: self.image_signatures = backup
                            return indx, match_threshold
            else:
                batch_match_list = []
                for id, value in self.image_signatures.items():
                    k = id
                    v = value['img_sign']
                    if np.array_equal(img_sign, v):
                        t = 0.0
                    else:
                        t = self.gis.normalized_distance(img_sign, v)
                    if t < match_threshold:
                        batch_match_list.append((k,t))
                if use_ssim:
                    batch_ssim_list = []
                    img = self.cvt_2_mosaic_gary(image_or_path)
                    for i in batch_match_list:
                        id = i[0]
                        score = calculate_ssim(img, self.image_signatures[id]['gary'])
                        if score > ssim_threshold:
                            batch_ssim_list.append((id, i[1], score))

                    if batch_ssim_list != []:
                        if match_image_signatures: self.image_signatures = backup
                        return batch_ssim_list
                    else: 
                        if batch_match_list != []: 
                            if match_image_signatures: self.image_signatures = backup
                            return [i + (None, ) for i in batch_match_list]
                        else:
                            print('SSIM未匹配到图像')
                            raise Exception("No match found")
                else:
                    if batch_match_list != []:
                        if match_image_signatures: self.image_signatures = backup
                        return batch_match_list
                    else: raise Exception("No match found")
                #else: return [k for k, v in self.image_signatures['img_sign'].items() if self.gis.normalized_distance(img_sign, v) < match_threshold]
        except Exception as e:
            print("Error matching image")
            if match_image_signatures: self.image_signatures = backup
            print(e)
        if match_image_signatures: self.image_signatures = backup

    def match_image_SSIM(self, image_or_path, ssim_threshold = 0.8, first_match = False, batch_match = False):
        if self.image_signatures == None:
            raise Exception("Image signatures not loaded")
        
        try:
            img = self.cvt_2_mosaic_gary(image_or_path)
            if batch_match == False:
                if first_match == True:
                    for id, value in self.image_signatures.items():
                        k = id
                        v = value['gary']
                        t = calculate_ssim(img, v)
                        if t > ssim_threshold:
                            return k, t
                    raise Exception("No match found")
                else:
                    indx = None
                    t = 0
                    for id, value in self.image_signatures.items():
                        k = id
                        v = value['gary']
                        t = calculate_ssim(img, v)
                        if t > ssim_threshold:
                            indx = k
                            ssim_threshold = t
                    if indx == None: raise Exception("No match found")
                    else: return indx,ssim_threshold
            else: 
                batch_ssim_list = []
                for id, value in self.image_signatures.items():
                    k = id
                    v = value['gary']
                    t = calculate_ssim(img, v)
                    if t > ssim_threshold:
                        batch_ssim_list.append((k, t))
                if batch_ssim_list != []:
                    return batch_ssim_list
                else: raise Exception("No match found")

        except Exception as e:
            print("Error matching image")
            print(e)       
    def cvt_2_mosaic_gary(self, img_or_path):
        if type(img_or_path) == str:
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path
        h, w = img.shape[:2]
        rate = h / 24
        new_h = int(h / rate)
        new_w = int(w / rate)
        img = cv2.resize(img, (new_w, new_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img


if __name__ == "__main__":
    
    #示例
    def list_all_files(directory):
        path_list = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                path_list.append(os.path.join(root, file))
        return path_list
    
    def get_parent_and_grandparent_dir(file_path):
        # 获取父文件夹路径
        parent_dir = os.path.dirname(file_path)
        # 获取父文件夹名称
        parent_name = os.path.basename(parent_dir)
        # 获取祖父文件夹路径
        grandparent_dir = os.path.dirname(parent_dir)
        # 获取祖父文件夹名称
        grandparent_name = os.path.basename(grandparent_dir)
        return parent_name, grandparent_name
    
    all_image_path = list_all_files('E:/zzz_frames_cut_new_01')
    all_path = [str(path) for path in all_image_path]

    IM = ImageMatcher()
    a = IM.gis.generate_signature(all_image_path[139])
    c = IM.gis.generate_signature('001410.jpg')
    b = IM.gis.normalized_distance(c, a)
    
    IM.process_image_to_story_block_signatures(all_path[:], use_ssim = True)
    IM.save_story_blocks_image_signatures('zzz_01_cut01.pkl','zzz_01_cut01.json')
    
    #IM.load_story_blocks_image_signatures('zzz_01.pkl', 'zzz_01.json')
    #IM.load_story_blocks_image_signatures('test1.pkl', 'test1.json')
    #IM.load_image_signatures('test1.pkl')
    ip = IM.story_blocks_image_signatures['index_path_dict']
    #d = IM.match_image(image_or_path = all_image_path[139],match_threshold = 0.6, use_ssim = True, ssim_threshold=0.35)
    
    #d = IM.block_match_image(image_path = all_image_path[139],match_threshold = 0.6, use_ssim = True, ssim_threshold=0.35)
    
    
    false = 0
    for index_, p in enumerate(all_image_path[444:1253]):
        #if index_%1000==0: print(f'{index_}/{len(all_image_path) - 1}')
        parent_dir, grandparent_dir = get_parent_and_grandparent_dir(p)
        path = f'{grandparent_dir}/{parent_dir}/{os.path.basename(p)}'
        start_time = time.time()
        matched = IM.block_match_image(image_or_path = p,match_threshold = 0.6, use_ssim = True, ssim_threshold=0.35)
        end_time = time.time()
        print(f"Matching took {end_time - start_time} seconds")
        if parent_dir == 'other' and matched == None:
            print(f'非剧情帧, {path}')

        if parent_dir == 'other' and matched != None:
            print(f'非剧情帧 误检, {path}, {ip[matched[0]]}')
            false+=1

        if parent_dir == 'story' and matched == None:
            print(f'漏检, {path}')
            false+=1
            
        if parent_dir == 'story' and matched != None:
            if ip[matched[0]] == path:
                print(f'剧情帧 正确, {path}')
            else:
                print(f'剧情帧 误检, {path}, {ip[matched[0]]}')
                false+=1

    print(false/len(all_image_path))
        


    
    """all_image_path = list_all_files('E:/zzz_frames')
    all_path = [str(path) for path in all_image_path]
    IM = ImageMatcher()
    IM.process_image_to_story_block_signatures(all_path[:], use_ssim = True)
    s = IM.story_blocks_image_signatures['story_block']
    id = IM.story_blocks_image_signatures['id_block_dict']
    ip = IM.story_blocks_image_signatures['index_path_dict']
    pi = IM.story_blocks_image_signatures['path_index_dict']
    IM.save_story_blocks_image_signatures('test1.pkl','test1.json')
    for i in s:
        print(i[0], i[-1], len(i))

    print()
    pp = '006/story/060900.jpg'

    for index_, p in enumerate(all_image_path):
        if index_%1000==0: print(f'{index_}/{len(all_image_path) - 1}')
        parent_dir, grandparent_dir = get_parent_and_grandparent_dir(p)
        path = f'{grandparent_dir}/{parent_dir}/{os.path.basename(p)}'
        find = False
        for i in s[id[pi[path]]]:
            if ip[i] == path:
                #print(ip[i],True)
                find = True
                break
        if find==False:
            print(path,False)"""

    """all_image_path = list_all_files(r'E:\nn480')#os.listdir(r'E:\nn4k-480-images\01')
    all_path = [str(path) for path in all_image_path]

    img_dict = {}
    for i, img in enumerate(all_path):
        img_dict[i] = img

    IM = ImageMatcher()

    #IM.process_image_to_signatures(all_path[:], use_ssim = True)
    #IM.save_image_signatures("test1.pkl")
    IM.load_image_signatures("test1.pkl")
    start_time = time.time()
    k = IM.match_image(r'000306.jpg', batch_match=False, match_threshold = 0.6, use_ssim=True)
    #k = IM.match_image_SSIM(r'QQ20241224125135.png', ssim_threshold = 0.8, first_match=False, batch_match=True)
    end_time = time.time()
    print(f"Matching took {end_time - start_time} seconds")
    if type(k) == list: 
        for kk in k:
            print(kk)
            #if str(img_dict[int(kk)])[-13:-11] == '26':
            print(img_dict[int(kk[0])])
        print(len(k))
        print(k)
    if type(k) == tuple: 
        print(img_dict[int(k[0])])
        print(k)""" 