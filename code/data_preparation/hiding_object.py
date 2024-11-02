import math
import numpy as np
import cv2
import PIL
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import scipy

def average_pooling_2d(image, pool_size, pool_stride, pool_process_func = None):
    """
    example:
    img = np.random.randint(100, size=(5, 5, 3))
    ap = average_pooling_2d( img, pool_size=(3, 3, 1), pool_stride = 1)
    """
    image_shape = np.array(image.shape)
    pool_size = np.array(pool_size)
    new_shape = tuple( (image_shape - pool_size) // pool_stride + 1 ) + tuple(pool_size)
    new_strides = tuple(np.array(image.strides) * pool_stride) + image.strides
    blocked = np.lib.stride_tricks.as_strided(image, shape=new_shape, strides=new_strides)    
    if pool_process_func is not None:
        blocked = pool_process_func(blocked)
    return np.mean(blocked, axis=tuple(range(image.ndim, blocked.ndim)))

def average_pooling_transparency_mask( transparency_mask ):
    ap = average_pooling_2d( transparency_mask, pool_size=(4, 4, 1), pool_stride = 4)
    ap = (ap > 0.2).astype(int)
    return ap

def show(img):
    plt.imshow(img.round().astype(int))
    plt.show()

def get_transparency_mask(img):    if len(img.shape) > 2 and img.shape[2] > 3:        alpha = img[:, :, 3:4] / 255.        alpha = (alpha > 0).astype(int)        return alpha    return np.ones(img.shape)

def preprocess_for_resnet(images):
    return preprocess_input( np.array(images) )

def read_image(path, standard_shape, resize_to_standard):    
    #img = cv2.imread( path, cv2.IMREAD_UNCHANGED )
    img = np.array( PIL.Image.open( path ) )

    if resize_to_standard:
        img = cv2.resize(img, (standard_shape[1], standard_shape[0]))
        #PIL.Image.resize     

    assert (list(img.shape)[:2] == list(standard_shape)[:2]), "Shape must be equal to {}, but it is {}".format( standard_shape, img.shape )
     
    #img = (cv2.cvtColor( img.astype(np.float32) / 255., cv2.COLOR_BGR2RGB ) * 255).astype(int)

    #show(img)
    return img

def read_images(paths, standard_shape, resize_to_standard):
    imgs = []
    for path in paths:
        img = read_image(path, standard_shape, resize_to_standard)
        img = img[:, :, :3]
        imgs.append( img )
    return imgs

def read_images_with_transparency_mask(paths, standard_shape, resize_to_standard):
    imgs = []
    transparency_masks = []
    for path in paths:
        img = read_image(path, standard_shape, resize_to_standard)
        transparency_mask = get_transparency_mask(img)
        
        img = img[:, :, :3]
        imgs.append( img * transparency_mask )

        transparency_mask = average_pooling_transparency_mask( transparency_mask )
        transparency_masks.append( transparency_mask )

    return imgs, transparency_masks

def pad_centric( array, reference_shape ):
    """
    https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros    
    """    
    result = np.zeros(reference_shape)
    y_pad = int(round((reference_shape[0] - array.shape[0]) / 2))
    x_pad = int(round((reference_shape[1] - array.shape[1]) / 2))    
    result[ y_pad : y_pad + array.shape[0], x_pad : x_pad + array.shape[1] ] = array
    return result

def euclidean_transform(img, rotation_angle, scale_factor, final_shape):
    """Rotate and scale
    https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    """
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)        
    M = cv2.getRotationMatrix2D((cX, cY), -rotation_angle, scale_factor)    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])        
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    img = cv2.warpAffine(img.astype(float), M, (nW, nH))
    if final_shape is not None:
        img = pad_centric( img, final_shape )
    #show(img)
    return img

def get_models(feature_layer_name, av_pool_size, av_pool_stride):
    model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling=None)
    resnet_output = model.get_layer(feature_layer_name).output

    bg_output = keras.layers.AveragePooling2D(pool_size=av_pool_size, strides=av_pool_stride, padding='valid')(resnet_output)

    obj_output = keras.layers.GlobalAveragePooling2D()(resnet_output)

    bg_model = keras.models.Model( inputs = model.input, outputs = bg_output )
    obj_model = keras.models.Model( inputs = model.input, outputs = obj_output )
    return bg_model, obj_model

def is_border(coordinates, bg_shape, obj_shape, border_width):
    return (coordinates[0] < border_width
            or coordinates[0] > bg_shape[0] - border_width - obj_shape[0]
            or coordinates[1] < border_width
            or coordinates[1] > bg_shape[1] - border_width - obj_shape[1])


class ObjectHider:
    """
    Utility for retrieving relative coordinates of an OBJ image inside a BG image.    
    """
    def __init__(self,
                bg_size = (960, 1300),
                obj_size = (80, 80),
                feature_layer = "conv2_block3_out",
                av_pool_stride = 1,
                local_search = True,
                min_scale_factor = 0.75,
                max_rotation_angle = 30,
                n_obj_candidates = 5,
                min_score = 0.7,
                border_width = 20,
                resize_to_standard = False      ):
        """
        bg_size : tuple
            (height, width) size of a BG image, default is (960, 1300);
        obj_size : tuple
            (height, width) size of an OBJ image, default is (80, 80);
        local_search : bool
            If True than a local search is done. Default is True;
            Local search: ```The second part of the algorithm is a local search. We use a similar procedure in a local area, found on the stage of the global search, but with a small stride (typically 1 pixel). During the local search, we also applied some rotation (from -30 degrees to +30 degrees) and down-scaling (with a factor of 1.0-0.75).```
        min_scale_factor : float
            Lower bound of a scale factor of an OBJ during a local search. Applied if local_search is True. Default is 0.75.
        max_rotation_angle : float
            Maximum angle (degrees) of OBJ rotation anticlockwise and clockwise. Applied if local_search is True. Default is 30. If it is 30, so, the range is [-30 ... 30].
        n_obj_candidates : int
            Maximum number of OBJs to hide in a BG.
        min_score : float
            Minimum score of OBJ relevance for an OBJ to be in a list of hiding in a BG.        
        """

        self.bg_size = bg_size
        self.obj_size = obj_size
        self.feature_layer = feature_layer
        self.av_pool_stride = av_pool_stride
        self.local_search = local_search
        self.min_scale_factor = min_scale_factor
        self.max_rotation_angle = max_rotation_angle
        self.n_obj_candidates = n_obj_candidates
        self.min_score = min_score
        self.border_width = border_width
        self.resize_to_standard = resize_to_standard

        self.map_scale_factor = 4  # Depends on a selected feature_layer !!!! CHECK!
        av_pool_size = ( int(round(self.obj_size[0] / self.map_scale_factor)), int(round(self.obj_size[1] / self.map_scale_factor)) )
        self.bg_model, self.obj_model = get_models(self.feature_layer, av_pool_size, self.av_pool_stride)

    def get_best_placement( self, bg_paths, obj_paths ):
        """
        Provides coordinates of OBJs (obj_paths) in BGs (bg_paths)

        Return : list of tuples
            Each item (tuple) of the list corresponds to one BG, which path is provided in bg_paths, with keeping an initial order.
            Item's structure: (obj_index, score, obj_coordinates)
                obj_index : index in the obj_paths list;
                score : a score (from 0 to 1) of relevance between the BG and the OBJ (with obj_index) in obj_coordinates;
                obj_coordinates : (y, x, rotation angle, scale factor),
                    where x and y are relative coordinated of an OBJ in a BG,
                    rotation angle is clock-wise rotation angle of an OBJ, in degrees,
                    scale factor is a scale factor of an OBJ (usually less than 1).
        """
        bg_imgs = read_images(bg_paths, self.bg_size, self.resize_to_standard)
        obj_imgs, transparency_masks = read_images_with_transparency_mask(obj_paths, self.obj_size, self.resize_to_standard)
        bg_preprocessed_imgs = preprocess_for_resnet( bg_imgs )
        obj_preprocessed_imgs = preprocess_for_resnet( obj_imgs )

        bg_tensor = self.bg_model.predict(bg_preprocessed_imgs)
        obj_tensor = self.obj_model.predict(obj_preprocessed_imgs)
        del bg_preprocessed_imgs, obj_preprocessed_imgs
        bg_tensor /= np.linalg.norm(bg_tensor, axis = -1, keepdims = True)
        obj_tensor /= np.linalg.norm(obj_tensor, axis = -1, keepdims = True)
        transparency_masks = np.array(transparency_masks)

        result = []

        for bg_index, bg_features in enumerate(bg_tensor):
            obj_indices, scores, coordinates = self.__global_search_one_bg(bg_features, obj_tensor, transparency_masks)
            bg_img = bg_imgs[bg_index]

            bg_result = []
            for index in range(len(obj_indices)):
                obj_index = obj_indices[index]
                obj_img = obj_imgs[index]
                obj_coordinates = coordinates[index]
                score = scores[index]
                angle, sf = 0., 1.                if self.local_search:                    obj_coordinates, angle, sf, score = self.__local_search(bg_img,                                                                            obj_img,                                                                            obj_coordinates )
                obj_coordinates = (obj_coordinates[0], obj_coordinates[1], angle, sf)
                bg_result.append( (obj_index, score, obj_coordinates) )
            result.append( bg_result )

        return result

    def __global_search_one_bg(self, bg_features, obj_tensor, transparency_masks):
        
        #similarity_maps = []
        #for obj_index, obj_features in enumerate(obj_tensor):
        #    transparency_mask = transparency_masks[obj_index]
        #    def multiply_transparency_mask(pool):
        #        return pool * transparency_mask
        #    tuned_bg_features = average_pooling_2d( bg_features, pool_size=(20, 20, 1), pool_stride = 1, pool_process_func = multiply_transparency_mask)            
        #    tuned_bg_features /= np.linalg.norm(tuned_bg_features, axis = -1, keepdims = True)  
        #        
        #    similarity_maps.append( np.tensordot( obj_features, tuned_bg_features, axes=((0), (2))) )
        #similarity_maps = np.array(similarity_maps)

        similarity_maps = np.tensordot( obj_tensor, bg_features, axes=((1), (2)))

        max_score_coordinates = [ np.unravel_index( map.argmax(), map.shape)  for map in similarity_maps ]
        max_scores = [ similarity_maps[i, coordinate[0], coordinate[1] ] for i, coordinate in enumerate(max_score_coordinates) ]

        best_obj_indices_unfiltered = np.argsort( max_scores )[ ::-1 ][ : (self.n_obj_candidates * 2) ]
        best_obj_indices = []
        best_coordinates = []
        best_scores = []
    
        for index in best_obj_indices_unfiltered:
            score = max_scores[index]
            if score < self.min_score:
                break
            coordinates = np.array(max_score_coordinates[index])
            coordinates *= self.map_scale_factor
            #coordinates += int(round(self.map_scale_factor / 2))
            coordinates = tuple(coordinates)

            # TODO: not ignore, but take next coordinate of the obj from the heat-map.
            if not is_border( coordinates, self.bg_size, self.obj_size, self.border_width ):
                best_coordinates.append( coordinates )
                best_obj_indices.append( index )
                best_scores.append( score )

        best_obj_indices = best_obj_indices[ : self.n_obj_candidates ]
        best_scores = best_scores[ : self.n_obj_candidates ]
        best_coordinates = best_coordinates[ : self.n_obj_candidates ]

        return best_obj_indices, best_scores, best_coordinates

    def __local_search(self, bg_img, obj_img, obj_coordinates):
        backuping_denom = 4.7 # <= 2 / (sqrt(2) - 1)
        #print(obj_coordinates)
        search_block_coordinates = (    int( obj_coordinates[0] - obj_img.shape[0] / backuping_denom ),                                        int( obj_coordinates[1] - obj_img.shape[1] / backuping_denom ),                                        int( obj_coordinates[0] + obj_img.shape[0] + obj_img.shape[0] / backuping_denom ),                                        int( obj_coordinates[1] + obj_img.shape[1] + obj_img.shape[1] / backuping_denom ) )        assert (    search_block_coordinates[0] >= 0                    and search_block_coordinates[2] <= bg_img.shape[0]                    and search_block_coordinates[1] >= 0                    and search_block_coordinates[3] <= bg_img.shape[1]  ), "Search block exceeds the border!"                search_block = bg_img [ search_block_coordinates[0] : search_block_coordinates[2],                                search_block_coordinates[1] : search_block_coordinates[3] ]        angle, sf, score = self.__get_best_transformation( search_block, obj_img   )        if sf < 1.:            obj_coordinates = ( obj_coordinates[0] + int(round( obj_img.shape[0] * (1./sf - 1.) / 2 )),
                                obj_coordinates[1] + int(round( obj_img.shape[1] * (1./sf - 1.) / 2 ))  )

        return obj_coordinates, angle, sf, score

    def __get_best_transformation(self, search_block, obj_img):
        best_score = 0.
        best_sf = 1.
        best_angle = 0
        best_coordinates = (0, 0)
        best_obj_img = obj_img

        sf_steps = int(round( (1. - self.min_scale_factor) / .025 + 1.))
        angle_steps = int(round( self.max_rotation_angle * 2 / 2.5 + 1.))
        
        batch = [ search_block ]
        scale_factors = np.linspace(self.min_scale_factor, 1.0, num = sf_steps)
        angles = np.linspace(-self.max_rotation_angle, self.max_rotation_angle, num = angle_steps)
        for sf in scale_factors:
            for angle in angles:
                transformed_obj_img = euclidean_transform(obj_img, angle, sf, search_block.shape)
                if transformed_obj_img.shape[0] > search_block.shape[0] or transformed_obj_img.shape[1] > search_block.shape[1]:
                    continue
                batch.append( transformed_obj_img )
        
        batch = preprocess_for_resnet(batch)
        batch_tensor = self.obj_model.predict( batch )
        del batch
        batch_tensor /= np.linalg.norm(batch_tensor, axis = -1, keepdims = True)

        scores = np.dot( batch_tensor[ 1: ], batch_tensor[0] )
        best_index = np.argmax( scores )

        best_sf_inx, best_angle_inx = np.unravel_index( best_index, (sf_steps, angle_steps) )
        best_sf = scale_factors[best_sf_inx]
        best_angle = angles[best_angle_inx]
        best_score = scores[best_index]
        
        return best_angle, best_sf, best_score
