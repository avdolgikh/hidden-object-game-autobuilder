import os
import numpy as np
import pandas as pd
import uuid
import cv2
from estimating_coordinates import ObjectCoordinatesEstimator
from drawing import Drawer
from image_similarities import *
from image_utils import *
import matplotlib.pyplot as plt
from time import perf_counter

BG_FOLDER = '../../data/third_stage_bgs'
OBJ_FOLDER = '../../data/third_stage_objs'

EXPERIMENT_RESULTS_FOLDER = '../../experiments/exp24_results'
EXPERIMENT_RESULTS_HEAT_MAPS = '../../experiments/exp24_results/heat_maps'
EXPERIMENT_RESULTS_META_FILE = '../../experiments/exp24_results_meta.csv'

def show(img):
    plt.imshow( cv2.cvtColor( img, cv2.COLOR_BGR2RGB) ) # , cmap='gray'
    plt.show()

def imread(path, alpha=False, grayscale=False):
    if alpha:
        return cv2.imread( path, cv2.IMREAD_UNCHANGED ).astype(np.float32)  / 255.
    if grayscale:
        return cv2.imread( path, 0 ).astype(np.float32)  / 255.
    return cv2.imread( path ).astype(np.float32)  / 255.

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
    with open(os.path.join(folder, '{}.png'.format(id)), 'wb') as file:
        fig.savefig(file, bbox_inches='tight')
        plt.close(fig)

def sample_hyper_params():
    smooth = False
    global_stride = 10 # np.random.choice([ 5, 10 ])
    global_stride = (global_stride, global_stride)
    make_score_heat_map = False
    global_similarity_func = np.random.choice([
                                                #gradient5_color5_similarity,
                                                #gradient7_color3_similarity,
                                                #gradient_color_geom_similarity,
                                                #resnet_level4_similarity,
                                                #resnet_level3_similarity,
                                                #resnet_level2_similarity,                                                
                                                #mimicry334_similarity,                                                
                                                #color30_similarity,
                                                #ecc_grad_similarity,
                                                #ssim_similarity,
                                                #mutual_information_similarity,
                                                resnet_2_similarity,
                                                resnet_3_2_similarity,
                                                resnet_3_similarity,
                                                resnet_4_2_similarity,
                                                resnet_4_4_similarity,
                                                resnet_4_similarity,
                                                resnet_5_2_similarity,
                                                resnet_5_similarity,                                                
                                            ])
    global_sampling_temperature = "max_score" # 0.000000001 # np.random.choice([0.1, 0.01, 0.001])
    local_search = True # np.random.choice([ True, False ])
    # local_similarity_func = np.random.choice([
    #                                             geometrical_similarity,
    #                                             mutual_information_similarity,
    #                                             ssim_similarity,
    #                                             gradient_similarity,
    #                                             ssd_similarity,
    #                                             abs_similarity,
    #                                             gradient5_color5_similarity,
    #                                             gradient7_color3_similarity,
    #                                             gradient_color_geom_similarity,
    #                                             resnet_level4_similarity,
    #                                             resnet_level3_similarity,
    #                                             resnet_level2_similarity,                                                
    #                                             mimicry334_similarity,
    #                                             ecc_grad_similarity,
    #                                         ])
    local_similarity_func = global_similarity_func
    scale_down_bg_height = 600 # np.random.choice([ 300, 600 ])

    return (    smooth, global_stride, make_score_heat_map,
                global_similarity_func, global_sampling_temperature,
                local_search, local_similarity_func, scale_down_bg_height )

def search_for_pair(    bg_img, obj_img, obj_path, bg_path,
                        bg_original_shape, obj_original_shape,
                        smooth, global_stride, make_score_heat_map,
                        global_similarity_func, global_sampling_temperature,
                        local_search, local_similarity_func, drawer,
                        result_folder, heatmaps_folder, scale_down_bg_height    ):

    print(global_similarity_func.__name__)

    time = perf_counter()
    estimator = ObjectCoordinatesEstimator( bg_img, obj_img, smooth, global_stride, make_score_heat_map,
                                            global_similarity_func, global_sampling_temperature, local_search,
                                            local_similarity_func, scale_down_bg_height )
    obj_coordinates, score = estimator.get_best_coordinates()
    time = perf_counter() - time

    meta = {    "score": np.round(score, 2),
                "coord": obj_coordinates,
                "gl_sim": global_similarity_func.__name__,
                "stride": global_stride,
                "tau": global_sampling_temperature,
                "obj": os.path.basename(obj_path),
                "bg_hgt": scale_down_bg_height,
            }
    if local_search:
        meta["l_sim"] = local_similarity_func.__name__
    
    fig = drawer.draw( bg_img, obj_img, obj_coordinates, meta )

    id = str(uuid.uuid4())
    save_result_img(fig, result_folder, "hog_" + id)

    meta["id"] = id
    meta["obj"] = obj_path
    meta["bg"] = bg_path    
    meta["bg_shape"] = bg_original_shape
    meta["obj_shape"] = obj_original_shape
    meta["bg_final_shape"] = bg_img.shape
    meta["obj_final_shape"] = obj_img.shape
    meta["time"] = np.round(time, 2)

    if make_score_heat_map:
        save_result_img(estimator.score_heatmap, heatmaps_folder, "hp_" + id)
        save_result_img(estimator.score_hist, heatmaps_folder, "hist_" + id)
    
    return meta

def stochastic_search_for_pair(bg_path, obj_path, drawer):
    bg_img = imread(bg_path, alpha=True, grayscale=False)
    obj_img = imread(obj_path, alpha=True, grayscale=False)

    bg_original_shape = bg_img.shape
    obj_original_shape = obj_img.shape

    print('bg_img shape:', bg_img.shape)
    print('obj_img shape:', obj_img.shape)

    (   smooth, global_stride, make_score_heat_map,
        global_similarity_func, global_sampling_temperature,
        local_search, local_similarity_func, scale_down_bg_height     ) = sample_hyper_params()

    return search_for_pair( bg_img, obj_img, obj_path, bg_path,
                            bg_original_shape, obj_original_shape,
                            smooth, global_stride, make_score_heat_map,
                            global_similarity_func, global_sampling_temperature,
                            local_search, local_similarity_func, drawer,
                            EXPERIMENT_RESULTS_FOLDER, EXPERIMENT_RESULTS_HEAT_MAPS, scale_down_bg_height     )



if __name__ == '__main__':
    
    bg_paths = read_paths(BG_FOLDER)
    obj_paths = read_paths(OBJ_FOLDER)

    print("bg_paths:", len(bg_paths))
    print("obj_paths:", len(obj_paths))
    
    drawer = Drawer()
    meta_data = []

    # Parallelize:
    for i in range(1000):
        print( "======================\n#{} iter:".format(i) )

        bg_path = np.random.choice(bg_paths)
        obj_path = np.random.choice(obj_paths)
        print('bg_path:', bg_path)
        print('obj_path:', obj_path)
        
        try:
            meta = stochastic_search_for_pair(bg_path, obj_path, drawer)
            meta_data.append( meta )

        except Exception as ex:
            print(ex)

        if i % 50 == 0:
            pd.DataFrame(meta_data).to_csv(path_or_buf=EXPERIMENT_RESULTS_META_FILE, header=True, index=False, encoding='utf-8')

    pd.DataFrame(meta_data).to_csv(path_or_buf=EXPERIMENT_RESULTS_META_FILE, header=True, index=False, encoding='utf-8')

        
        
    
                

