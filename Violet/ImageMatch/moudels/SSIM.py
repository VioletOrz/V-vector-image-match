import os
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def calculate_ssim(img_or_path1, img_or_path2):
    """
    计算两张图像的 SSIM 值。
    """
    if type(img_or_path1) == str:
        img_or_path1 = cv2.imread(img_or_path1)
    if type(img_or_path2) == str:
        img_or_path2 = cv2.imread(img_or_path2)
        
    # 如果图像不是灰度图，转换为灰度图 
    if len(img_or_path1.shape) == 3:
        img_or_path1 = cv2.cvtColor(img_or_path1, cv2.COLOR_BGR2GRAY)
    if len(img_or_path2.shape) == 3:
        img_or_path2 = cv2.cvtColor(img_or_path2, cv2.COLOR_BGR2GRAY)
    
    # 调整尺寸
    if img_or_path1.shape != img_or_path2.shape:
        img_or_path2 = cv2.resize(img_or_path2, (img_or_path1.shape[1], img_or_path1.shape[0]))

    # 计算 SSIM

    score, _ = ssim(img_or_path1, img_or_path2, full=True)

    return score

def calculate_ssim_for_batch(batch, img, image_signatures, ssim_threshold, is_batch = False):
    """
    在给定的批次中查找与输入图像最匹配的图像。

    Args:
        img: 输入图像 (可能是灰度图像)。
        batch: 一个列表，其中每个元素是一个元组 (id, time)。
        image_signatures: 一个字典，键是图像 ID，值是包含 'gary' 键的字典，'gary' 键对应图像的灰度签名。
        ssim_threshold: SSIM 分数的阈值。
        is_batch: 一个布尔值，指示是否以批处理模式运行。

    Returns:
        如果 is_batch 为 False: 返回一个元组 (max_score, match_id, match_time)。
            max_score: 最高 SSIM 分数。
            match_id: 最高分数的图像 ID。
            match_time: 最高分数的图像时间。
        如果 is_batch 为 True: 返回一个列表，其中每个元素都是一个元组 (score, match_id, match_time)，
            包含所有大于阈值的匹配。
        如果没有找到匹配，返回 (0, None, None) 如果 is_batch 为 False 或返回空列表如果 is_batch 为 True。
    """
    
   
    if is_batch:
        matches = []
        for i in batch:
            id = i[0]
            score = calculate_ssim(img, image_signatures[id])
            if score > ssim_threshold:
                matches.append((id, i[1], score))
        return matches
    else:
        max_score = 0
        match_id = None
        match_time = None
        for i in batch:
            id = i[0]
            score = calculate_ssim(img, image_signatures[id])
            if score > max_score and score > ssim_threshold:
                max_score = score
                match_id = id
                match_time = i[1]


        return match_id, match_time, max_score

def process_in_threads(batch_match_list, img, image_signatures, ssim_threshold, is_batch = False, num_threads=4):
    """
    使用多线程计算 SSIM，并获取最大值。
    
    :param batch_match_list: 匹配列表
    :param img: 当前图像
    :param image_signatures: 包含签名的字典
    :param ssim_threshold: SSIM 阈值
    :param num_threads: 线程数量
    :return: (max_ssim, match_id, match_time)
    """
    max_ssim = 0
    match_id = None
    match_time = None
    
    # 将数据分成 num_threads 份
    batch_size = len(batch_match_list) // num_threads + 1
    batches = [batch_match_list[i:i + batch_size] for i in range(0, len(batch_match_list), batch_size)]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        #start_time = time.time()
        future_to_batch = {
            executor.submit(calculate_ssim_for_batch, batch, img, image_signatures, ssim_threshold, is_batch): batch
            for batch in batches
        }
        
        
        if is_batch:
            all_matches = []
            for future in as_completed(future_to_batch):
                 matches = future.result()
                 all_matches.extend(matches)
            

        else:
            for future in as_completed(future_to_batch):
                temp_id, temp_time, max_score = future.result()
                if max_score > max_ssim:
                    max_ssim = max_score
                    match_id = temp_id
                    match_time = temp_time
            
    if is_batch:
        return all_matches
    else:
        #end_time = time.time()
        #print(f"Thread pool created in {end_time - start_time} seconds")
        return match_id, match_time, max_ssim