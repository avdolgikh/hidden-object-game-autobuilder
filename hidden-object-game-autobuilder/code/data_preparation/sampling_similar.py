import numpy as np
import skimage.measure
from scipy.special import softmax

def sampling_max(score_map):    return np.unravel_index( score_map.argmax(), score_map.shape)def sampling_boltzman_with_maxpooling(score_map):    maxpooled_map = max_pooling(score_map)    #print("maxpooled_map.shape:", maxpooled_map.shape)    maxpooled_coordinates = sampling_boltzman(maxpooled_map) # make several samples.    return maxpooled_coordinates[0]*10, maxpooled_coordinates[1]*15 # TODO: find the best form region# TODO: make stride (10,15) as a parameterdef max_pooling(map):    # https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy    
    return skimage.measure.block_reduce(map, (10,15), np.max)def sampling_boltzman(map, temperature = 0.01):    map_shape = map.shape    map = map.flatten()

    try:
        # the bigger temperature (e.g. 10) the more uniformish
        # small temperature  (0.0001) means get max (not sampling)
        probabilities = softmax( map / temperature)
        sampled_index = np.random.choice( len(map), p = probabilities)        return np.unravel_index( sampled_index, map_shape)            

    except Exception as ex:
        print(ex)
        return sampling_max(map)
    