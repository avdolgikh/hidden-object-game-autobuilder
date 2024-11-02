import os
import numpy as np
import pandas as pd
import cv2
from drawing import Drawer
from image_similarities import *
from search_pipeline import *

RESULTS_FOLDER = '../../experiments/demo_results'

if __name__ == '__main__':

    bg_path = '../../data/hog backgrounds/jungle mysteries/bg_2.jpg'
    obj_path = '../../data/hog images/daily hidden object/H_2_Folding_fan.png'

    drawer = Drawer()

    bg_img = imread(bg_path, alpha=True, grayscale=False)
    obj_img = imread(obj_path, alpha=True, grayscale=False)

    print('bg_img shape:', bg_img.shape)    print('obj_img shape:', obj_img.shape)

    scale_down_bg_height = 300
    smooth = False
    global_stride = 30
    global_stride = (global_stride, global_stride)
    make_score_heat_map = False
    global_similarity_func = resnet_level5_similarity
    global_sampling_temperature = 0.01
    local_search = True
    local_similarity_func = abs_similarity

    for _ in range(10):
        search_for_pair(    bg_img, obj_img, obj_path, bg_path,
                            bg_img.shape, obj_img.shape,
                            smooth, global_stride, make_score_heat_map,
                            global_similarity_func, global_sampling_temperature,
                            local_search, local_similarity_func, drawer,
                            RESULTS_FOLDER, None, scale_down_bg_height     )
    
                

