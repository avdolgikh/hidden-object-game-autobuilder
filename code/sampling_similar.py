import numpy as np
import skimage.measure
from scipy.special import softmax

def sampling_max(score_map):
    return skimage.measure.block_reduce(map, (10,15), np.max)

    try:
        # the bigger temperature (e.g. 10) the more uniformish
        # small temperature  (0.0001) means get max (not sampling)
        probabilities = softmax( map / temperature)
        sampled_index = np.random.choice( len(map), p = probabilities)

    except Exception as ex:
        print(ex)
        return sampling_max(map)
    