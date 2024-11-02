import cv2
import numpy as np

def filter_by_alpha_channel(img):
        return img[:, :, :3] * img[:, :, 3:4]
def resize(img, height):
    aspectRatio = img.shape[1] / img.shape[0]
    area = height * (height * aspectRatio)
    height = np.sqrt(area / aspectRatio)
    width = height * aspectRatio
    
    img = cv2.resize(img, (int(width), int(height)))
    return img
        img = filter_by_alpha_channel(img)
        image_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        return image_lab.reshape((-1, 3)), img.shape
        segmented = cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2BGR)
# https://en.wikipedia.org/wiki/Color_quantization
def quantize_colors(img, n_classes = 10):
    #img = image.astype(np.float32)  / 255. # check if it is necessary
    img = filter_by_alpha_channel(img)
    image_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    data = image_lab.reshape((-1, 3))
    labels, centers = kmeans(data, n_classes)    
    segmented_lab = centers[labels.flatten()].reshape(img.shape)
    segmented = cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2BGR)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels, centers = cv2.kmeans( data , n_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)    
    img = np.uint8(img * 255)
        if hierarchy[0][i][3] == -1:            
# https://www.learnopencv.com/wp-content/uploads/2015/07/motion-models.jpg

#def scale(img):    
#    shape = ( int(img.shape[1] * sf), int(img.shape[0] * sf) )
#    return cv2.resize(img, shape)

def euclidean_transform(img, rotation_angle, scale_factor):
    """Rotate and scale
    https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    """
    # center
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)        
    M = cv2.getRotationMatrix2D((cX, cY), -rotation_angle, scale_factor)
    # !!! TODO: take scale factor into account !!!!
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])        
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(img, M, (nW, nH))

# how to restrict big rotation cv2.findTransformECC?
# TODO: Also try searching in gradient space!
# https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
def transform_ecc(search_block, obj_img):
    # Convert images to grayscale
    search_block = cv2.cvtColor(search_block, cv2.COLOR_BGR2GRAY)
    obj_img_gray = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)

    # Define the motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY # cv2.MOTION_EUCLIDEAN # cv2.MOTION_AFFINE # cv2.MOTION_HOMOGRAPHY # cv2.MOTION_TRANSLATION # cv2.MOTION_HOMOGRAPHY

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 1000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-20

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#findtransformecc
    # Run the ECC algorithm. The results are stored in warp_matrix
    #(cc, warp_matrix) = cv2.findTransformECC( np.float32(search_block[:,:,:3]), np.float32(obj_img[:,:,:3]), warp_matrix, warp_mode, criteria, None, 1 )
    (cc, warp_matrix) = cv2.findTransformECC( search_block, obj_img_gray, warp_matrix, warp_mode, criteria, None, 1 )
    # !!!! inputMask - transparency_mask !!??

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        obj_img_aligned = cv2.warpPerspective (obj_img, warp_matrix, (search_block.shape[1], search_block.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        obj_img_aligned = cv2.warpAffine( obj_img, warp_matrix, (search_block.shape[1], search_block.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return obj_img_aligned

def get_histogram_1d(img):
    hist, _ = np.histogram(img, bins=20)
    return np.array(hist) / len(img) 

    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    #grad = np.concatenate( [dx, dy], axis=-1 )

    # https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    grad = cv2.addWeighted( np.absolute(dx), 0.5, np.absolute(dy), 0.5, 0)

    return grad
    new_obj_height = int( bg_img.shape[0] / bg_obj_ratio )
    if new_obj_height != obj_img.shape[0]:
        obj_img = resize(obj_img,  new_obj_height)

    #new_bg_height = int( obj_img.shape[0] * bg_obj_ratio )    
    #if new_bg_height >= bg_img.shape[0]:
    #    obj_img = resize(obj_img, int( bg_img.shape[0] / bg_obj_ratio ) )
    #else:
    #    bg_img = resize(bg_img, new_bg_height )
    return bg_img, obj_img
