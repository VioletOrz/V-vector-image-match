from Violet.ImageMatch.moudels.match import ImageMatcher
from Violet.ImageMatch.moudels.cosin import cosine_similarity
from image_match.goldberg import ImageSignature
from Violet.Violet_base import list_all_files, write_pkl_file, read_pkl_file, get_parent_and_grandparent_dir,read_json_file,list_immediate_subfolders
import os
import random
import time
import cv2

from Violet.ImageMatch.moudels.search_tree_match import Map_matcher
from search_tree_updata import updata_database_from_G_data_pkl, updata_database_from_img


if __name__ == '__main__':
    "E:/Genshin_frames_circle/01"
    "Genshin_01.pkl"
    """save_name = './pkl/test_story'#'E:/MapMatch/pkl/test_12k'
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
    mm.min_distance = mm.min_distance"""

    all_path = list_immediate_subfolders('E:/Genshin_Impact_story/')

    all_img_path = []
    for path in all_path[24:44]:
        all_img_path += list_all_files(path)
        
    #ll_img_path = all_img_path[:10]
    updata_database_from_img(all_img_path, './pkl', 'Genshin_5.1')
    #updata_database_from_img('./pkl/Genshin_5.0.pkl', './pkl', 'Genshin_5.0')
    
    danmaku = read_json_file('1.json')

    all_path = list_all_files('E:/Genshin_Impact/')

    
    for p in all_path:
        r = mm.search_img_topk('002160.jpg')
        #start_time = time.time()

        #mm.insert(cv2.imread('000165.jpg'))
        #end_time = time.time()
        #print(f"Processing took {end_time - start_time} seconds")
        print(r)
        if r[0] == None:
            print(r)
        
        #print(danmaku[r[0]])

        #time.sleep(5)
    

    err = 0
    for d in mm.G_data:
        min_2 = 1
        for f in mm.fa_node:
            dis = mm.gis.normalized_distance(d['vector'], mm.G_data[f]['vector'])
            if dis < min_2:
                min_2 = dis
        if min_2 > (mm.min_distance * 2): 
            print(d['id'])
            err+=1
    print(f'{err}/{len(mm.G_data)}')

    min_1 = 1
    for i in mm.fa_node:
        for j in mm.fa_node:
            if i != j:
                d = mm.gis.normalized_distance(mm.G_data[i]['vector'], mm.G_data[j]['vector'])
                if d == None: continue
                if d < min_1:
                    min_1 = d
                    print(d)
    print(min_1,mm.min_distance)