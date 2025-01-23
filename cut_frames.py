from Violet.ImageMatch.moudels.SSIM import calculate_ssim
from Violet.ImageMatch.moudels.match import ImageMatcher
import cv2
import os
import numpy as np

def list_all_files(directory):
    path_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            path_list.append(os.path.join(root, file))
    return path_list

def get_subfolders(folder_path):
    """
    获取指定文件夹中的所有子文件夹路径。
    
    :param folder_path: 要扫描的文件夹路径
    :return: 子文件夹路径列表
    """
    subfolders = []
    
    # 使用 os.scandir 高效遍历文件夹
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if entry.is_dir():  # 检查是否为文件夹
                subfolders.append(entry.path)
    
    return subfolders

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

def cut_frames_new(output_dir):

    IM =ImageMatcher()
    IM.load_story_blocks_image_signatures('zzz_01.pkl', 'zzz_01.json')
    ip = IM.story_blocks_image_signatures['index_path_dict']
    pi = IM.story_blocks_image_signatures['path_index_dict']

    start_index = 0

    all_img_path = list_all_files('E:/zzz_frames')

    tmp_img = '001327'

    all_files = get_subfolders('E:/zzz_frames')
    tmp_list = []
    for p in all_files:
        ttmp = get_subfolders(p)
        for pp in ttmp:
            tmp_list.append(pp)
    frames_list = [list_all_files(i) for i in tmp_list]
    
    for all_path in frames_list:
        for i, path in enumerate(all_path):
            f, g = get_parent_and_grandparent_dir(path)
            p = g + '/' + f + '/' + os.path.basename(path)
            index = pi[p]
            if i < start_index: continue
            print(f'{i}/{len(all_path)}')
            
            output = output_dir + '/' + ip[index]
            if os.path.exists(output[:-10]) == False:
                os.makedirs(output[:-10])
            if i == start_index:
                print(f'保存到： {output}')
                img = cv2.imread('E:/zzz_frames/' + ip[index])
                cv2.imwrite(output, img)
                tmp_img = index
                continue
            s = IM.image_signatures[index]['img_sign']#IM.gis.generate_signature(img)
            if IM.gis.normalized_distance(s, IM.image_signatures[tmp_img]['img_sign']) < 0.20 or np.array_equal(IM.image_signatures[tmp_img]['img_sign'], s):
                print('跳过')
                continue
            else:
                print(f'保存到： {output}')
                img = cv2.imread('E:/zzz_frames/' + ip[index])
                cv2.imwrite(output, img)
                tmp_img = index

def cut_frames(frame_file_path, output_dir):
    
    
    all_path = list_all_files(frame_file_path)
    parent_name, grandparent_name = get_parent_and_grandparent_dir(frame_file_path)
    name = os.path.basename(frame_file_path)

    output = output_dir + '/' + parent_name + '/' + name

    if os.path.exists(output) == False:
        os.makedirs(output)

    all_img = []
    for p in all_path:
        all_img.append(cv2.imread(p))
    
    tmp_img = 0
    for i in range(0, len(all_img)):
        print(f'正在处理： {all_path[i]}')
        img_name = os.path.basename(all_path[i])
        if i == 0:
            
            cv2.imwrite(output + '/' + img_name, all_img[i])
            tmp_img = i
            continue

        ssim = calculate_ssim(all_img[tmp_img], all_img[i])
        

        if ssim >= 0.85:
            print(f'ssim: {ssim} 跳过')
            pass
        else:
            tmp_output = output + '/' + img_name
            print(f'ssim: {ssim} 保存图像到 {tmp_output}')
            cv2.imwrite(tmp_output, all_img[i])
            tmp_img = i

#def cut_frames_cosin(pkl_path,img_path):


if __name__ == '__main__':
    frame_path = 'E:/zzz_frames'
    #cut_frames_new('E:/zzz_frames_cut_new_02')
    """all_part_path = get_subfolders(frame_path)

    all_frames_path = []
    for p in all_part_path:
        tmp = get_subfolders(p)
        for t in tmp:
            all_frames_path.append(t)
    
    start_index = 9
    end_index = 10
    for i, p in enumerate(all_frames_path):
        index = i + 1
        if index < start_index: continue
        cut_frames(p, 'E:/zzz_frames_cut')
    pass"""
    
    #while True: pass

    