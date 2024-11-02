import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL
import uuid
from random import shuffle

from hiding_object import ObjectHider

import sys
try:
    sys.path.insert(0, '../prototype')
except Exception as ex:
    print(ex)

from drawing import Drawer
from search_pipeline import search_for_pair
from image_similarities import *


BG_FOLDER = '../../data/third_stage_bgs'
OBJ_FOLDER = '../../data/third_stage_objs'

EXPERIMENT_RESULTS_FOLDER = '../../experiments/exp30_results'
EXPERIMENT_RESULTS_HEAT_MAPS = '../../experiments/exp30_results/heat_maps'
EXPERIMENT_RESULTS_META_FILE = '../../experiments/exp30_results_meta.csv'

TOOL_RESULTS_FOLDER = '../../experiments/final_tool_results'

def read_paths(folder):
    paths = []
    for subfolder in os.listdir(folder):
        subfolder_path = folder + "/" + subfolder
        if os.path.isfile(subfolder_path):
            continue
        for filename in os.listdir(subfolder_path):
            file_path = subfolder_path + "/" + filename
            if os.path.isfile(file_path):
                paths.append( file_path )
    return paths


def save_result_img(fig, folder, id):
    if id is None:
        id = str(uuid.uuid4())
    with open(os.path.join(folder, '{}.png'.format(id)), 'wb') as file:
        fig.savefig(file, bbox_inches='tight')
        plt.close(fig)

def read_image(path):
    #img = cv2.imread( path, cv2.IMREAD_UNCHANGED )
    img = cv2.imread( path, cv2.IMREAD_UNCHANGED ).astype(np.float32)  / 255.
    #img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
    #img = np.array( PIL.Image.open( path ) )
    return img 

def run_benchmark(bg_img, obj_img, obj_path, bg_path, drawer):
    smooth = False
    global_stride = (4, 4)
    make_score_heat_map = False
    global_similarity_func = resnet_2_similarity
    global_sampling_temperature = "max_score"
    local_search = True
    local_similarity_func = resnet_2_similarity
    scale_down_bg_height = None # 960

    search_for_pair(bg_img, obj_img, obj_path, bg_path,
                    bg_img.shape, obj_img.shape,
                    smooth, global_stride, make_score_heat_map,
                    global_similarity_func, global_sampling_temperature,
                    local_search, local_similarity_func, drawer,
                    EXPERIMENT_RESULTS_FOLDER, EXPERIMENT_RESULTS_HEAT_MAPS, scale_down_bg_height     )




if __name__ == '__main__':
    bg_paths = read_paths(BG_FOLDER)
    obj_paths = read_paths(OBJ_FOLDER)

    shuffle(bg_paths)
    shuffle(obj_paths)

    bg_paths = bg_paths[:5]

    print("bg_paths:", len(bg_paths))
    print("obj_paths:", len(obj_paths))
    
    hider = ObjectHider(    n_obj_candidates = 3,
                            local_search = True,                            
                            min_score = 0.8,
                            resize_to_standard = True,
                            av_pool_stride = 1,
                            border_width = 20  )
    
    output = hider.get_best_placement( bg_paths, obj_paths )
    
    drawer = Drawer()
    
    for bg_index, bg_meta in enumerate(output):
        bg_path = bg_paths[bg_index]
        bg_img = read_image( bg_path )
        bg_img = cv2.resize( bg_img, (1300, 960) )
    
        for obj_meta in bg_meta:
            obj_index, score, obj_coordinates = obj_meta
            obj_path = obj_paths[obj_index]
            obj_img = read_image( obj_path )
            obj_img = cv2.resize( obj_img, (80, 80) )
    
            # Benchmark:
            #run_benchmark(bg_img, obj_img, obj_path, bg_path, drawer)
    
            # The Tool:
            meta = {
                        "score": np.round(score, 4),
                        "coord": obj_coordinates,
                        "obj": os.path.basename(obj_path),
                   }
            fig = drawer.draw( bg_img, obj_img, obj_coordinates, meta )
            save_result_img(fig, TOOL_RESULTS_FOLDER, None)

    
