import numpy as npimport scipyimport cv2import skimagefrom resnet50_utils import *from image_utils import *# TODO: add unit tests (like similar ones have similarity=1, etc.)# ============================ Pixel-based similarities ==========================# For pixel-based, we can blur images... Can it make things better?# img1, img2 = smooth(img1, img2)# ~L1
def abs_similarity(img1, img2, transparency_mask):
    #img1, img2 = smooth(img1, img2)
    distance = np.sum( np.abs( img1 - img2 ) * transparency_mask )
    # normalization:
    distance /= np.sum( transparency_mask )
    return 1. - distance

# ~L2
# sum of squared differences (SSD) function (sum of residual errors), https://courses.cs.washington.edu/courses/cse576/05sp/papers/MSR-TR-2004-92.pdf
def ssd_similarity(img1, img2, transparency_mask):
    distance = np.sqrt( np.sum( ( (img1 - img2) * transparency_mask )**2 ) )
    # normalization:
    distance /= np.sqrt( np.sum( transparency_mask**2 ) )
    return 1. - distance

# Lq (Minkowski)
def minkowski_similarity(img1, img2, transparency_mask, p = 5):
    distance = ( np.sum( ( np.abs(img1 - img2) * transparency_mask )**p ) )**(1./p)
    # normalization:
    distance /= np.sum( np.abs(transparency_mask)**p )**(1./p)
    return 1. - distance

# ~cosine similarity
# Normalized cross-correlation, https://courses.cs.washington.edu/courses/cse576/05sp/papers/MSR-TR-2004-92.pdf
def correlation_similarity(img1, img2, transparency_mask):
    """
    Note, however, that the NCC score is undefined if either of the two patches has zero variance
    (and in fact, its performance degrades for noisy low-contrast regions).
    """ 
    def get_unbiased(img):
        img = img * transparency_mask
        mean_img = np.sum( img ) / np.sum( transparency_mask )
        return img - mean_img
    
    img1_unbiased = get_unbiased(img1)
    img2_unbiased = get_unbiased(img2)
    
    score = np.sum( img1_unbiased * img2_unbiased )
    score /= np.sqrt( np.sum( ( img1_unbiased )**2 ) )
    score /= np.sqrt( np.sum( ( img2_unbiased )**2 ) )

    return score


# ================== Distribution similarities =====================

# Earth Mover's Distance
# or Wasserstein distance
# https://stats.stackexchange.com/questions/404775/calculate-earth-movers-distance-for-two-grayscale-images
# https://gist.github.com/duhaime/211365edaddf7ff89c0a36d9f3f7956c
# https://stackoverrun.com/ru/q/10432044
# https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html
# https://arxiv.org/pdf/1509.02237.pdf
# https://theailearner.com/tag/wasserstein-metric-opencv-python/
#cv2.cv.CalcEMD2()
def wasserstein_similarity(img1, img2, transparency_mask):    
    n_channels = img1.shape[2] if len(img1.shape) > 2 else 1
    distance = np.mean( [   wasserstein_distance_one_channel(   img1[:, :, channel],
                                                                img2[:, :, channel],
                                                                transparency_mask[:, :, channel] ) 
                            for channel in range(n_channels) ])    
    return 1. / (1. + distance)

def wasserstein_distance_one_channel(img1, img2, transparency_mask):
    img1 = np.uint8(img1 * 255)
    img2 = np.uint8(img2 * 255)

    visible = transparency_mask > 0
    img1 = img1[visible].reshape(-1)
    img2 = img2[visible].reshape(-1)

    img1_hist = get_histogram_1d(img1)
    img2_hist = get_histogram_1d(img2)

    distance = scipy.stats.wasserstein_distance(img1_hist, img2_hist)
    return distance


# https://arxiv.org/ftp/arxiv/papers/1712/1712.07540.pdf
# Mutual information
def mutual_information_similarity(img1, img2, transparency_mask):
    n_channels = img1.shape[2] if len(img1.shape) > 2 else 1
    mi = np.mean( [ mutual_information_similarity_one_channel(  img1[:, :, channel],
                                                                img2[:, :, channel],
                                                                transparency_mask[:, :, channel] ) 
                    for channel in range(n_channels) ])
    return mi

# https://matthew-brett.github.io/teaching/mutual_information.html
def mutual_information_similarity_one_channel(img1, img2, transparency_mask):
    visible = transparency_mask > 0
    img1 = img1[visible].reshape(-1)
    img2 = img2[visible].reshape(-1)

    hist_2d, x_edges, y_edges = np.histogram2d(img1, img2, bins=20) #256?

    # Convert bins counts to probability values
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals    
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

    # normalization?    
    return mi

# ============================ Geometrical similarities ====================# How to utilize transparency_mask?def geometrical_similarity(img1, img2, transparency_mask):
    set1 = get_external_contour(img1 * transparency_mask)    set2 = get_external_contour(img2 * transparency_mask)        if len(set1) == 0 or len(set2) == 0:        return 0.    
    h1 = scipy.spatial.distance.directed_hausdorff(set1, set2)[0]
    h2 = scipy.spatial.distance.directed_hausdorff(set2, set1)[0]
    distance = max(h1, h2)        return 1. / (1. + distance)# Check! Visualize!def keypoints_similarity(img1, img2, transparency_mask):
    set1 = cv2.goodFeaturesToTrack( cv2.cvtColor(img1 * transparency_mask, cv2.COLOR_BGR2GRAY), 100, 0.05, 10)    set2 = cv2.goodFeaturesToTrack( cv2.cvtColor(img2 * transparency_mask, cv2.COLOR_BGR2GRAY), 100, 0.05, 10)        if len(set1) == 0 or len(set2) == 0:        return 0.    set1 = set1.reshape(-1, 2)    set2 = set2.reshape(-1, 2)    
    h1 = scipy.spatial.distance.directed_hausdorff(set1, set2)[0]
    h2 = scipy.spatial.distance.directed_hausdorff(set2, set1)[0]
    distance = max(h1, h2)        return 1. / (1. + distance)# ============================ Semantic similarities ===========================#resnet50_level5, resnet50_level4, resnet50_level3, resnet50_level2 = get_resnet50_img_embeddings_model()
rn_stage2_output, \
rn_stage3_2_output, \
rn_stage3_output, \
rn_stage4_2_output, \
rn_stage4_4_output, \
rn_stage4_output, \
rn_stage5_2_output, \
rn_stage5_output = get_resnet50_img_embeddings_model()

def resnet_similarity(model, img1, img2, transparency_mask):
    img1, img2 = img1*transparency_mask, img2*transparency_mask
    img_data = preprocess_for_resnet([img1, img2])
    embeddings = model.predict( img_data )
    return resnet_embeddings_similarity( embeddings[0], embeddings[1] )

def resnet_2_similarity(img1, img2, transparency_mask):
    return resnet_similarity(rn_stage2_output, img1, img2, transparency_mask)

def resnet_3_2_similarity(img1, img2, transparency_mask):
    return resnet_similarity(rn_stage3_2_output, img1, img2, transparency_mask)

def resnet_3_similarity(img1, img2, transparency_mask):
    return resnet_similarity(rn_stage3_output, img1, img2, transparency_mask)

def resnet_4_2_similarity(img1, img2, transparency_mask):
    return resnet_similarity(rn_stage4_2_output, img1, img2, transparency_mask)

def resnet_4_4_similarity(img1, img2, transparency_mask):
    return resnet_similarity(rn_stage4_4_output, img1, img2, transparency_mask)

def resnet_4_similarity(img1, img2, transparency_mask):
    return resnet_similarity(rn_stage4_output, img1, img2, transparency_mask)

def resnet_5_2_similarity(img1, img2, transparency_mask):
    return resnet_similarity(rn_stage5_2_output, img1, img2, transparency_mask)

def resnet_5_similarity(img1, img2, transparency_mask):
    return resnet_similarity(rn_stage5_output, img1, img2, transparency_mask)

#def resnet_level5_similarity(img1, img2, transparency_mask):
#    img1, img2 = img1*transparency_mask, img2*transparency_mask
#    img_data = preprocess_for_resnet([img1, img2])
#    embeddings = resnet50_level5.predict( img_data )
#    return resnet_embeddings_similarity( embeddings[0], embeddings[1] )
#
#def resnet_level4_similarity(img1, img2, transparency_mask):
#    img1, img2 = img1*transparency_mask, img2*transparency_mask
#    img_data = preprocess_for_resnet([img1, img2])
#    embeddings = resnet50_level4.predict( img_data )
#    return resnet_embeddings_similarity( embeddings[0], embeddings[1] )
##def resnet_level3_similarity(img1, img2, transparency_mask):
#    img1, img2 = img1*transparency_mask, img2*transparency_mask
#    img_data = preprocess_for_resnet([img1, img2])
#    embeddings = resnet50_level3.predict( img_data )
#    return resnet_embeddings_similarity( embeddings[0], embeddings[1] )
#
#def resnet_level2_similarity(img1, img2, transparency_mask):
#    img1, img2 = img1*transparency_mask, img2*transparency_mask
#    img_data = preprocess_for_resnet([img1, img2])
#    embeddings = resnet50_level2.predict( img_data )
#    return resnet_embeddings_similarity( embeddings[0], embeddings[1] )
# ============================ Mixed similarities ===========================def gradient5_color5_similarity(img1, img2, transparency_mask):    color = abs_similarity(img1, img2, transparency_mask)    gradient = gradient_similarity(img1, img2, transparency_mask)    return .5 * gradient + .5 * colordef gradient7_color3_similarity(img1, img2, transparency_mask):    color = abs_similarity(img1, img2, transparency_mask)    gradient = gradient_similarity(img1, img2, transparency_mask)    return .7 * gradient + .3 * colordef gradient3_color7_similarity(img1, img2, transparency_mask):    color = abs_similarity(img1, img2, transparency_mask)    gradient = gradient_similarity(img1, img2, transparency_mask)    return .3 * gradient + .7 * colordef gradient_color_geom_similarity(img1, img2, transparency_mask):    color = abs_similarity(img1, img2, transparency_mask)    gradient = gradient_similarity(img1, img2, transparency_mask)    geometrical = geometrical_similarity(img1, img2, transparency_mask)    return .4 * gradient + .3 * color + .3 * geometricaldef mimicry343_similarity(img1, img2, transparency_mask):    geometrical = geometrical_similarity(img1, img2, transparency_mask)    color = abs_similarity(img1, img2, transparency_mask)    mi = mutual_information_similarity(img1, img2, transparency_mask)    return .3 * geometrical + .4 * color + .3 * midef mimicry334_similarity(img1, img2, transparency_mask):    geometrical = geometrical_similarity(img1, img2, transparency_mask)    color = abs_similarity(img1, img2, transparency_mask)    mi = mutual_information_similarity(img1, img2, transparency_mask)    return .3 * geometrical + .3 * color + .4 * midef mimicry523_similarity(img1, img2, transparency_mask):    geometrical = geometrical_similarity(img1, img2, transparency_mask)    color = abs_similarity(img1, img2, transparency_mask)    mi = mutual_information_similarity(img1, img2, transparency_mask)    return .5 * geometrical + .2 * color + .3 * midef mimicry325_similarity(img1, img2, transparency_mask):    geometrical = geometrical_similarity(img1, img2, transparency_mask)    color = abs_similarity(img1, img2, transparency_mask)    mi = mutual_information_similarity(img1, img2, transparency_mask)    return .3 * geometrical + .2 * color + .5 * mi# ============================ Misc similarities ===========================# Color-similarity (based on color quantization)
def color10_similarity(img1, img2, transparency_mask):
    img1, img2 = reduce_colors_jointly(img1, img2, n_classes=10)
    distance = np.abs( np.sum(img1*transparency_mask) - np.sum(img2*transparency_mask) ) / np.sum( transparency_mask )
    return 1. - distance

def color20_similarity(img1, img2, transparency_mask):
    img1, img2 = reduce_colors_jointly(img1, img2, n_classes=20)
    distance = np.abs( np.sum(img1*transparency_mask) - np.sum(img2*transparency_mask) ) / np.sum( transparency_mask )
    return 1. - distance

def color30_similarity(img1, img2, transparency_mask):
    img1, img2 = reduce_colors_jointly(img1, img2, n_classes=30)
    distance = np.abs( np.sum(img1*transparency_mask) - np.sum(img2*transparency_mask) ) / np.sum( transparency_mask )
    return 1. - distance
# Enhanced Correlation Coefficient
# http://xanthippi.ceid.upatras.gr/people/evangelidis/george_files/PAMI_2008.pdf
def ecc_similarity(img1, img2, transparency_mask):
    n_channels = img1.shape[2] if len(img1.shape) > 2 else 1
    coef = np.mean( [   cv2.computeECC( templateImage = img1[:,:,channel],
                                        inputImage = img2[:,:,channel]
                                        ,inputMask = np.uint8(transparency_mask[:,:,channel])
                                       ) 
                        for channel in range(n_channels) ]) * .5 + .5
    return coef
def ecc_grad_similarity(img1, img2, transparency_mask):
    img1 = get_gradient(img1 * transparency_mask)
    img2 = get_gradient(img2 * transparency_mask)

    n_channels = img1.shape[2] if len(img1.shape) > 2 else 1
    coef = np.mean( [   cv2.computeECC( templateImage = img1[:,:,channel],
                                        inputImage = img2[:,:,channel]                                        
                                       ) 
                        for channel in range(n_channels) ]) * .5 + .5
    return coefdef ecc_gray_grad_similarity(img1, img2, transparency_mask):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) * transparency_mask[:,:,0]
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) * transparency_mask[:,:,0]

    img1 = get_gradient(img1)
    img2 = get_gradient(img2)
    
    coef = cv2.computeECC(  templateImage = img1,
                            inputImage = img2
                            ) * .5 + .5
    return coef
def gradient_similarity(img1, img2, transparency_mask):
    img1 = get_gradient(img1 * transparency_mask)
    img2 = get_gradient(img2 * transparency_mask)
    distance = np.sum( np.abs( img1 - img2 ) )
    
    # normalization:
    distance /= np.sum( transparency_mask )

    return 1. - distance
# MAD, https://courses.cs.washington.edu/courses/cse576/05sp/papers/MSR-TR-2004-92.pdf, p.17

# TODO: look at source. Find custom support of multichannel?
# Structural! Based on statistics?
def ssim_similarity(img1, img2, transparency_mask):
    #img1, img2 = img1*transparency_mask, img2*transparency_mask
    similarity = skimage.measure.compare_ssim(img1, img2, multichannel=True)
    return similarity

# https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
# https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html
def farneback_similarity(img1, img2, transparency_mask):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) * transparency_mask[:,:,0]
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) * transparency_mask[:,:,0]

    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    distance = np.sum( np.abs(flow) ) # / np.sum( transparency_mask ) / 2
    
    return 1. - distance

# Based on key points (opencv_3_computer_vision_with_python_cookbook.pdf, p.217)
# The idea: compare coordinates of keypoints somehow.
    # The closer they the more similar images in some sense
    #pts1 = cv2.goodFeaturesToTrack(img1, 30, 0.05, 10)
    #pts2 = cv2.goodFeaturesToTrack(img2, 30, 0.05, 10)



