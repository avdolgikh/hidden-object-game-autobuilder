import numpy as np
import cv2
import matplotlib.pyplot as plt
from image_utils import euclidean_transform


class Drawer (object):

    def draw(self, bg_img, obj_img, obj_coordinates, meta={}):

        if obj_coordinates[2] != 0 or obj_coordinates[3] != 1:
            obj_img = euclidean_transform(obj_img, obj_coordinates[2], obj_coordinates[3])

        if len(obj_img.shape) > 2 and obj_img.shape[2] > 3:
            img = self.__overlay_with_alpha(bg_img, obj_img, obj_coordinates)        else:            img = self.__overlay(bg_img, obj_img, obj_coordinates)        figure = plt.figure(figsize=(14, 13), dpi=250)        plt.imshow( cv2.cvtColor( img, cv2.COLOR_BGR2RGB) )
        
        #plt.title("result")

        legend_loc = self.__get_legend_loc(bg_img.shape[1], bg_img.shape[0], obj_coordinates[1], obj_coordinates[0])
        self.__show_meta(meta, legend_loc)
        #plt.show()
        return figure

    def __show_meta(self, meta, legend_loc):
        label = ""
        delim = ""
        for item in meta:
            label += "{}{}: {}".format( delim, item, meta[item] )
            delim = "\n"

        if len(label) > 0:
            plt.plot([], [], ' ', label=label)
            plt.legend(loc = legend_loc)

    def __get_legend_loc(self, width, height, x, y):
        horiz = "left" if x > width/2 else "right"
        vert = "upper" if y > height/2 else "lower"
        return "{} {}".format(vert, horiz)            

    def __overlay(self, bg_img, obj_img, obj_coordinates):
        result_img = np.copy(bg_img)        
        result_img[ obj_coordinates[0] : obj_coordinates[0] + obj_img.shape[0],                    obj_coordinates[1] : obj_coordinates[1] + obj_img.shape[1] ] = obj_img
        return result_img

    #https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
    def __overlay_with_alpha(self, bg_img, obj_img, obj_coordinates):
        obj_alpha = obj_img[:, :, 3]
        bg_alpha = 1.0 - obj_alpha
        # alpha channel is 0 for transparent parts

        result_img = np.copy(bg_img)

        for c in range(3):
            result_img[ obj_coordinates[0] : obj_coordinates[0] + obj_img.shape[0],                        obj_coordinates[1] : obj_coordinates[1] + obj_img.shape[1],
                        c ] = (obj_alpha * obj_img[:, :, c] + bg_alpha * bg_img[ obj_coordinates[0] : obj_coordinates[0] + obj_img.shape[0],                                                                                 obj_coordinates[1] : obj_coordinates[1] + obj_img.shape[1],                                                                                 c ])
        return result_img
