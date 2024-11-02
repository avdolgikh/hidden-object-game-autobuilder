import osimport mathimport numpy as npimport cv2import matplotlib.pyplot as plt
from image_utils import *
from image_similarities import geometrical_similarity

def show_gradient(image, dx, dy):
    plt.figure(figsize=(8,3))
    plt.subplot(131)
    plt.axis('off')
    plt.title('image')
    plt.imshow(image, cmap='gray')
    plt.subplot(132)
    plt.axis('off')
    plt.imshow(dx, cmap='gray')
    plt.title(r'$\frac{dI}{dx}$')
    plt.subplot(133)
    plt.axis('off')
    plt.title(r'$\frac{dI}{dy}$')
    plt.imshow(dy, cmap='gray')
    plt.tight_layout()
    plt.show()

def show_filtered(image, filtered):
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.axis('off')
    plt.title('image')
    plt.imshow(image[:, :, [2, 1, 0]])
    plt.subplot(122)
    plt.axis('off')
    plt.title('filtered')
    plt.imshow(filtered[:, :, [2, 1, 0]])
    plt.tight_layout(True)
    plt.show()

def show_kernel(image, kernel, filtered):
    plt.figure(figsize=(8,3))
    plt.subplot(131)
    plt.axis('off')
    plt.title('image')
    plt.imshow(image, cmap='gray')
    plt.subplot(132)
    plt.title('kernel')
    plt.imshow(kernel, cmap='gray')
    plt.subplot(133)
    plt.axis('off')
    plt.title('filtered')
    plt.imshow(filtered, cmap='gray')
    plt.tight_layout()
    plt.show()

def show_fft(fft_shift):
    #fft_shift = np.fft.fftshift(fft, axes=[0, 1])
    magnitude = cv2.magnitude(fft_shift[:, :, 0], fft_shift[:, :, 1])
    magnitude = np.log(magnitude)
    plt.axis('off')
    plt.imshow(magnitude, cmap='gray')
    plt.tight_layout()
    plt.show()

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

def show_with_countours(obj_paths):
    path = np.random.choice(obj_paths)
    image = cv2.imread( path, cv2.IMREAD_UNCHANGED ).astype(np.float32)  / 255.
    image = filter_by_alpha_channel(image)
    #plt.imshow( cv2.cvtColor( image, cv2.COLOR_BGR2RGB) )    #plt.show()    
    image_contour = draw_external_contour( image )    
    plt.imshow( cv2.cvtColor( image_contour, cv2.COLOR_BGR2RGB) )    plt.show()
    return image


if __name__ == '__main__':

    #image = np.full((480, 640, 3), 255, np.uint8)
    #image = np.full((480, 640, 3), (0, 0, 255), np.uint8)


    #path = r'..\..\data\Lena.png'
    #path = r'..\..\data\candle.png'

    obj_folder = '../../data/hog images'
    obj_paths = read_paths(obj_folder)

    for _ in range(30):
        image1 = show_with_countours(obj_paths)
        image2 = show_with_countours(obj_paths)
        print(geometrical_similarity( image1, image2, get_transparency_mask(image1)))

    
    #image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    #data = image_lab.reshape((-1, 3))
    #num_classes = 4
    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    #_, labels, centers = cv2.kmeans( data , num_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    #segmented_lab = centers[labels.flatten()].reshape(image.shape)
    #image = cv2.cvtColor(segmented_lab, cv2.COLOR_Lab2RGB)
    #
    #plt.imshow( image )    #plt.show()



    ## discrete Fourier transform:
    #fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
    ##show_fft(fft)    #    #fft_shift = np.fft.fftshift(fft, axes=[0, 1])    #    ## !!! Set the amplitudes for high frequencies to zero, leaving the others untouched    #sz = 5
    ##mask = np.zeros(fft_shift.shape, np.uint8)
    ##mask[ mask.shape[0] // 2 - sz : mask.shape[0] // 2 + sz, mask.shape[1] // 2 - sz : mask.shape[1] // 2 + sz, :] = 1
    #
    #mask = np.ones(fft_shift.shape, np.uint8)
    #mask[ mask.shape[0] // 2 - sz : mask.shape[0] // 2 + sz, mask.shape[1] // 2 - sz : mask.shape[1] // 2 + sz, :] = 0
    #
    #fft_shift *= mask    ##show_fft(fft_shift)    #    #print(mask.shape)    #    #fft = np.fft.ifftshift(fft_shift, axes=[0, 1])    #    #image = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)    ##image = mask[:, :, 0] * 255    
    #kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, cv2.CV_32F)
    #kernel /= math.sqrt((kernel * kernel).sum())
    #filtered = cv2.filter2D(image, -1, kernel)
    #show_kernel(image, kernel, filtered)

    #KSIZE = 11
    #ALPHA = 2
    #kernel = cv2.getGaussianKernel(KSIZE, 0)
    #kernel = -ALPHA * kernel @ kernel.T
    #kernel[KSIZE//2, KSIZE//2] += 1 + ALPHA
    #filtered = cv2.filter2D(image, -1, kernel)
    #show_filtered(image, filtered)


    #image = (image + 0.3 * np.random.rand(*image.shape).astype(np.float32)).clip(0, 1)

    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
    #image = cv2.GaussianBlur(image, (9, 9), 1.4)    #image = cv2.medianBlur((image * 255).astype(np.uint8), 7)    #image = cv2.bilateralFilter(image, -1, 0.3, 10)
    #dx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    #dy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    #show_gradient(image, dx, dy)    
    
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #image[..., 2] = cv2.equalizeHist(image[..., 2])
    #image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    
    #hist, bins = np.histogram(image, 256, [0, 255])
    #plt.fill(hist)
    #plt.xlabel('pixel value')
    #plt.show()

    #image = cv2.equalizeHist(image)

    #hist, bins = np.histogram(image, 256, [0, 255])
    #plt.fill_between(range(256), hist, 0)
    #plt.xlabel('pixel value')
    #plt.show()

    #image -= image.mean()
    #image /= image.std()
    #image = cv2.meanStdDev(image)

    #gamma = 1.22
    #image = np.power(image, gamma)

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print('Converted to grayscale')

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #print('Converted to HSV')

    #image[:, :, 2] *= 2
    #image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    #print('Converted back to BGR from HSV')

    #print('Shape:', image.shape)
    #print('Data type:', image.dtype)
    

    #image[:, :, [0, 2]] = image[:, :, [2, 0]]
    #image[:, :, [0, 2]] = image[:, :, [2, 0]]
    #image[:, :, 0] = (image[:, :, 0] * 0.9).clip(0, 1)
    #image[:, :, 1] = (image[:, :, 1] * 1.1).clip(0, 1)


    #image = image.astype(np.float32) / 255.
    #print('Shape:', image.shape)
    #print('Data type:', image.dtype)

    #image = np.clip(image*2, 0, 1)    
    #image = (image * 255).astype(np.uint8)
    #print('Shape:', image.shape)
    #print('Data type:', image.dtype)

    #cv2.imshow('', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()        #plt.imshow( cv2.cvtColor( image, cv2.COLOR_BGR2RGB) )    #plt.show()