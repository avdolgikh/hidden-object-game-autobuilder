import cv2
import numpy as np

def filter_by_alpha_channel(img):    if len(img.shape) > 2 and img.shape[2] > 3:
        return img[:, :, :3] * img[:, :, 3:4]    return imgdef get_transparency_mask(img):    return np.repeat( img[:, :, 3:4], 3, axis=2) if len(img.shape) > 2 and img.shape[2] > 3 else np.ones(img.shape)# https://stackoverflow.com/questions/33701929/how-to-resize-an-image-in-python-while-retaining-aspect-ratio-given-a-target-s/33702454
def resize(img, height):
    aspectRatio = img.shape[1] / img.shape[0]
    area = height * (height * aspectRatio)
    height = np.sqrt(area / aspectRatio)
    width = height * aspectRatio
    
    img = cv2.resize(img, (int(width), int(height)))
    return imgdef smooth(bg_img, obj_img):    # remove noise - smooth - to look at "bigger" features?        # Gaussian kernels        # Or working in frequency-space (Fourier transform)    bg_img = cv2.GaussianBlur(bg_img, (9, 9), 0)    obj_img = cv2.GaussianBlur(obj_img, (9, 9), 0)    return bg_img, obj_imgdef reduce_colors(bg_img, obj_img):    bg_img = quantize_colors(bg_img, n_classes = 15)    obj_img = quantize_colors(obj_img, n_classes = 5)    return bg_img, obj_imgdef reduce_colors_jointly(bg_img, obj_img, n_classes = 15):    bg_img, obj_img = np.copy(bg_img), np.copy(obj_img)    def to_data(img):        #img = img.astype(np.float32)  / 255. # check if it is necessary
        img = filter_by_alpha_channel(img)
        image_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        return image_lab.reshape((-1, 3)), img.shape    bg_img_data, bg_img_shape = to_data(bg_img)    obj_img_data, obj_img_shape = to_data(obj_img)    labels, centers = kmeans(np.concatenate( (bg_img_data, obj_img_data) ), n_classes = n_classes)    def to_image(labels, centers, img_shape):        segmented_lab = centers[labels.flatten()].reshape(img_shape)
        segmented = cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2BGR)        return segmented #* 255    bg_img[:, :, :3] = to_image(labels[ : len(bg_img_data) ], centers, bg_img_shape)    obj_img[:, :, :3] = to_image(labels[ len(bg_img_data) : ], centers, obj_img_shape)        return bg_img, obj_img
# https://en.wikipedia.org/wiki/Color_quantization
def quantize_colors(img, n_classes = 10):
    #img = image.astype(np.float32)  / 255. # check if it is necessary
    img = filter_by_alpha_channel(img)
    image_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    data = image_lab.reshape((-1, 3))
    labels, centers = kmeans(data, n_classes)    
    segmented_lab = centers[labels.flatten()].reshape(img.shape)
    segmented = cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2BGR)    image[:, :, :3] = segmented #* 255    return imagedef kmeans(data, n_classes):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels, centers = cv2.kmeans( data , n_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)        return labels, centers# https://stackoverflow.com/questions/21482534/how-to-use-shape-distance-and-common-interfaces-to-find-hausdorff-distance-in-op# https://www.tutorialspoint.com/find-and-draw-contours-using-opencv-in-python# https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html# https://www.thepythoncode.com/article/contour-detection-opencv-pythondef draw_external_contour(img):    external_contour = get_external_contour(img)    #print(external_contour.shape)    if len(external_contour) > 0:        img = np.uint8(img * 255)        cv2.drawContours(img, [external_contour], 0, (0, 255, 0), 1)        img = img.astype(np.float32)  / 255.    return imgdef get_external_contour(img):
    img = np.uint8(img * 255)    edges = cv2.Canny(img, 300, 200)    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #cv2.CHAIN_APPROX_NONE    external_contour = []    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:                        external_contour += contours[i].reshape(-1, 2).tolist()        return np.array( external_contour )def get_edged_binary_image(img):    img = np.uint8(img * 255)    img = cv2.Canny(img, 300, 200).astype(np.float32)  / 255.    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #otsu_thr, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)    #_, img = cv2.threshold(img, 100, 255, 0)    #_, img = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY_INV)    return img
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
def get_gradient(image):
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    #grad = np.concatenate( [dx, dy], axis=-1 )

    # https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    grad = cv2.addWeighted( np.absolute(dx), 0.5, np.absolute(dy), 0.5, 0)

    return graddef scale_images(bg_img, obj_img, bg_obj_ratio):
    new_obj_height = int( bg_img.shape[0] / bg_obj_ratio )
    if new_obj_height != obj_img.shape[0]:
        obj_img = resize(obj_img,  new_obj_height)

    #new_bg_height = int( obj_img.shape[0] * bg_obj_ratio )    
    #if new_bg_height >= bg_img.shape[0]:
    #    obj_img = resize(obj_img, int( bg_img.shape[0] / bg_obj_ratio ) )
    #else:
    #    bg_img = resize(bg_img, new_bg_height )
    return bg_img, obj_img
