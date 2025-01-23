from Violet.ImageMatch.moudels.match import ImageMatcher
from Violet.ImageMatch.moudels.cosin import cosine_similarity
from image_match.goldberg import ImageSignature
from Violet.Violet_base import list_all_files, write_pkl_file, read_pkl_file, get_parent_and_grandparent_dir,read_json_file
import os
import random
import time

from Violet.ImageMatch.moudels.search_tree_match import Map_matcher

def updata_database_from_img(img_files_path, pkl_file, save_name):
    """
    从图像文件更新数据库信息。

    该函数通过图像处理和数据处理，将图像中的地图信息转化为结构化数据，并进行一系列处理和保存操作，
    以便后续使用和查询。

    参数:
    img_files_path (str): 图像文件的路径，用于地图图像处理。
    pkl_file (str): 存储处理结果的文件路径，通常为一个目录。
    save_name (str): 保存处理结果的文件名前缀，用于生成多个相关的数据文件。
    """
    # 初始化地图匹配器对象
    mm = Map_matcher()

    start_time = time.time()
    # 将图像处理为结构化数据（pkl格式）
    mm.process_img_2_pkl(img_files_path, pkl_file + '/' + save_name + '.pkl')
    # 以下行代码被注释掉，如果需要从已处理的pkl文件中加载数据，可以使用它
    #mm.load_pkl(save_name + '.pkl')

    end_time = time.time()
    print(f"Processing took {end_time - start_time} seconds")
    start_time = time.time()

    # 使用贪婪算法获取最远点集，参数为点之间的最小距离阈值
    mm.greedy_furthest_points(int(len(mm.G_data)**0.5)) #001

    # 处理自由访问节点地图，建立节点之间的连接关系
    mm.process_fa_node_map() #002

    # 保存处理后的自由访问地图数据到指定路径
    mm.save_fa_map(pkl_file + '/' + save_name + '_fa_map.pkl')
    # 以下行代码被注释掉，如果需要从已处理的自由访问地图文件中加载数据，可以使用它
    #mm.load_fa_map(save_name + '_fa_map.pkl')

    # 打印自由访问节点信息，用于调试和验证处理结果
    print(len(mm.fa_node), mm.fa_node)

    # 生成搜索树，用于优化后续的数据查询和处理
    mm.generate_search_tree() #003

    # 保存生成的搜索树数据到指定路径
    mm.save_search_tree(pkl_file + '/' + save_name + '_search_tree.pkl')
    # 以下行代码被注释掉，如果需要从已处理的搜索树文件中加载数据，可以使用它
    #mm.load_search_tree(save_name + '_search_tree.pkl')
    end_time = time.time()
    print(f"Processing took {end_time - start_time} seconds")

def updata_database_from_G_data_pkl(G_data_pkl_path, pkl_file, save_name):
    """
    从G数据的pkl文件更新数据库。

    此函数的目的是通过给定的G数据pkl文件来更新数据库信息。它会进行一系列的数据处理步骤，
    包括地图匹配、数据点处理、生成搜索树等，最终将处理结果保存为新的pkl文件。

    参数:
    G_data_pkl_path (str): G数据pkl文件的路径。这是输入数据的来源。
    pkl_file (str): 用于保存处理结果的文件路径。这决定了输出文件存放的位置。
    save_name (str): 保存文件的名称前缀。这将用于生成输出文件的名称。

    返回:
    无
    """
    
    # 初始化地图匹配器
    mm = Map_matcher()
    
    # 加载G数据pkl文件
    mm.load_pkl(G_data_pkl_path)
    
    # 通过贪婪算法选择最远的点，可能代表选择30个最远的点
    start_time = time.time()

    mm.greedy_furthest_points(int(len(mm.G_data)**0.5)) #001

    # 处理自由访问节点地图，建立节点之间的连接关系
    mm.process_fa_node_map() #002

    # 保存处理后的自由访问地图数据到指定路径
    mm.save_fa_map(pkl_file + '/' + save_name + '_fa_map.pkl')
    # 以下行代码被注释掉，如果需要从已处理的自由访问地图文件中加载数据，可以使用它
    #mm.load_fa_map(save_name + '_fa_map.pkl')

    # 打印自由访问节点信息，用于调试和验证处理结果
    print(len(mm.fa_node), mm.fa_node)

    # 生成搜索树，用于优化后续的数据查询和处理
    mm.generate_search_tree() #003

    # 保存生成的搜索树数据到指定路径
    mm.save_search_tree(pkl_file + '/' + save_name + '_search_tree.pkl')
    # 以下行代码被注释掉，如果需要从已处理的搜索树文件中加载数据，可以使用它
    #mm.load_search_tree(save_name + '_search_tree.pkl')
    end_time = time.time()
    print(f"Processing took {end_time - start_time} seconds")



if __name__ == '__main__':

    G_data_pkl_path = r'pkl\test_12k.pkl'
    pkl_file = './pkl'
    save_name = 'test_12k'
    img_files_path = 'E:\Genshin_frames'
    
    #updata_database_from_img(img_files_path, pkl_file, save_name)
    #end_time = time.time()
    #print(f"Processing took {end_time - start_time} seconds")
    updata_database_from_G_data_pkl(G_data_pkl_path, pkl_file, save_name)