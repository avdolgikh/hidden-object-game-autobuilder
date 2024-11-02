import numpy as npimport cv2from image_similarities import *from visualizing_information import *from sampling_similar import *
from image_utils import *

class ObjectCoordinatesEstimator (object):
    def __init__(self, bg_img, obj_img, smooth, global_stride, make_score_heat_map,
                    global_similarity_func, global_sampling_temperature, local_search,
                    local_similarity_func, scale_down_bg_height):
        self.bg_img = bg_img
        self.obj_img = obj_img
        self.smooth = smooth
        self.global_stride = global_stride
        self.make_score_heat_map = make_score_heat_map
        self.global_similarity_func = global_similarity_func
        self.global_sampling_temperature = global_sampling_temperature
        self.local_search  = local_search
        self.local_similarity_func = local_similarity_func
        self.score_heatmap = None
        self.score_hist = None
        self.scale_down_bg_height = scale_down_bg_height
        self.scale_down = self.scale_down_bg_height is not None

    def get_best_coordinates(self):
        """
        returns (x, y, angle, scale_factor), score
        """        self.__preprocess()                stride = self.global_stride        global_score_map = self.__build_score_map(  self.bg_img,                                                    self.obj_img,                                                    self.global_similarity_func,                                                    stride=stride )        if self.make_score_heat_map:            self.__visualize_score_map(global_score_map)                coordinates = sampling_max(global_score_map)        #coordinates = sampling_boltzman(global_score_map, self.global_sampling_temperature)        score = global_score_map[coordinates]        coordinates = [ coordinates[0] * stride[0], coordinates[1] * stride[1] ]                angle, sf = 0., 1.        if self.local_search:            coordinates, angle, sf, score, self.obj_img = self.__local_search(  self.bg_img,                                                                                self.obj_img,                                                                                coordinates,                                                                                self.local_similarity_func  )
        coordinates = [ coordinates[0], coordinates[1], angle, sf ]
       
        self.__postprocess(coordinates)
        #print(score)
        
        return coordinates, score

    def __preprocess(self):        if self.scale_down:
            self.bg_preprocessing_scalefactor = self.bg_img.shape[0] / self.scale_down_bg_height
            self.bg_img = resize(self.bg_img, self.scale_down_bg_height)            self.obj_img = resize(self.obj_img, int(self.obj_img.shape[0] / self.bg_preprocessing_scalefactor))        self.bg_obj_ratio = self.bg_img.shape[0] / self.obj_img.shape[0]        if self.smooth:            self.bg_img, self.obj_img = smooth(self.bg_img, self.obj_img)        

    def __postprocess(self, coordinates):
        #print(coordinates)
        if self.scale_down:
            coordinates[0] = int(coordinates[0] * self.bg_preprocessing_scalefactor)
            coordinates[1] = int(coordinates[1] * self.bg_preprocessing_scalefactor)            
            #print(coordinates)

    def __build_score_map(self, bg_img, obj_img, similarity_func, stride=(1, 1)):
        score_map = np.zeros(   (   (bg_img.shape[0] - obj_img.shape[0]) // stride[0] + 1,
                                    (bg_img.shape[1] - obj_img.shape[1]) // stride[1] + 1   )  )

        transparency_mask = get_transparency_mask(obj_img)
        obj_img = filter_by_alpha_channel(obj_img)
        bg_img = filter_by_alpha_channel(bg_img)

        for i in range( score_map.shape[0] ):
          for j in range( score_map.shape[1] ):
            bg_part = bg_img[   i*stride[0] : i*stride[0] + obj_img.shape[0],
                                j*stride[1] : j*stride[1] + obj_img.shape[1] ]
            score_map[i, j] = similarity_func(bg_part, obj_img, transparency_mask)
        return score_map

    def __visualize_score_map(self, score_map):
        print("score_map.shape:", score_map.shape)                self.score_heatmap = show_heatmap(score_map)        self.score_hist = show_hist(score_map)

    def __local_search(self, bg_img, obj_img, best_global_coordinates, similarity_func):
        search_block_coordinates = (best_global_coordinates[0] - obj_img.shape[0] // 3, best_global_coordinates[1] - obj_img.shape[1] // 3)                search_block_coordinates = ( np.max([0, search_block_coordinates[0]]), np.max([0, search_block_coordinates[1]]) )        search_block = bg_img [ search_block_coordinates[0] : best_global_coordinates[0] + obj_img.shape[0] + obj_img.shape[0] // 3,                                search_block_coordinates[1] : best_global_coordinates[1] + obj_img.shape[1] + obj_img.shape[1] // 3 ]        local_coordinates, angle, sf, score, obj_img = self.__get_best_transformation( search_block,                                                                            obj_img,                                                                            similarity_func   )        global_coordinates = (  search_block_coordinates[0] + local_coordinates[0],
                                search_block_coordinates[1] + local_coordinates[1]  )
        return global_coordinates, angle, sf, score, obj_img

    def __get_best_transformation(self, search_block, obj_img, similarity_func):
        best_score = 0.
        best_sf = 1.
        best_angle = 0
        best_coordinates = (0, 0)
        best_obj_img = obj_img
        
        for sf in [1., 0.975, 0.95, 0.925, 0.9, 0.875, 0.85, 0.825, 0.8, 0.775, 0.75]:
            for angle in [-30, -27.5, -25, -22.5 -20, -17.5, -15, -12.5, -10, -7.5, -5, -2.5, 0,
                            2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30]:
                transformed_obj_img = euclidean_transform(obj_img, angle, sf)
                if transformed_obj_img.shape[0] > search_block.shape[0] or transformed_obj_img.shape[1] > search_block.shape[1]:
                    continue
                score_map = self.__build_score_map(search_block, transformed_obj_img, similarity_func)
                coordinates = sampling_max(score_map)
                if best_score < score_map[coordinates]:
                    best_score = score_map[coordinates]
                    best_angle = angle
                    best_sf = sf
                    best_coordinates = coordinates
                    best_obj_img = transformed_obj_img

        return best_coordinates, best_angle, best_sf, best_score, best_obj_img

    def __get_ecc_transformation(self, search_block, obj_img):
        best_score = 0.
        best_sf = 1.
        best_angle = 0
        best_coordinates = (0, 0)
        best_obj_img = transform_ecc(search_block, obj_img)
        return best_coordinates, best_angle, best_sf, best_score, best_obj_img
    
    
