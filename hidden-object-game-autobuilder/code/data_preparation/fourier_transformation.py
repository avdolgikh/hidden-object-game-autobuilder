import numpy as npimport cv2
  
def fft(x):    return np.fft.fftn( x.astype(complex) )    #return cv2.dft( x, flags=cv2.DFT_COMPLEX_OUTPUT )def ifft(x):    return np.absolute(np.fft.ifftn(x))    #return cv2.idft(x, flags=cv2.DFT_REAL_OUTPUT)

# https://courses.cs.washington.edu/courses/cse576/05sp/papers/MSR-TR-2004-92.pdf, p.20
# Phase correlation (Normalized cross-correlation using Fourier transformation)
def __build_fourier_neg_correlation_map(self, bg_img, obj_img):
    """
    bg_img, obj_img : images (grayscale or color or color with alpha)

    Note from https://courses.cs.washington.edu/courses/cse576/05sp/papers/MSR-TR-2004-92.pdf, p.21:
    Phase correlation has a reputation in some quarters of outperforming regular correlation, but
    this behavior depends on the characteristics of the signals and noise. If the original images are
    contaminated by noise in a narrow frequency band (e.g., low-frequency noise or peaked frequency
    'hum'), the whitening process effectively de-emphasizes the noise in these regions. However, if
    the original signals have very low signal-to-noise ratio at some frequencies (say, two blurry or lowtextured images with lots of high-frequency noise), the whitening process can actually decrease
    performance. To my knowledge, no systematic comparison of the two approaches (together with
    other Fourier-based techniques such as windowed and/or bias-gain corrected differencing) has been
    performed, so this is an interesting area for further exploration
    """        

    transparency_mask = get_transparency_mask(obj_img)
    obj_img = filter_by_alpha_channel(obj_img)
    bg_img = filter_by_alpha_channel(bg_img)
    obj_img = obj_img * transparency_mask

    obj_img_shape = obj_img.shape

    augmented_obj_img = np.zeros( bg_img.shape )
    augmented_obj_img[ : obj_img.shape[0],  : obj_img.shape[1] ] = obj_img
    obj_img = augmented_obj_img
        
    bg_fft = fft( bg_img )  
    obj_fft = fft( obj_img )

    map = np.multiply( np.conjugate(obj_fft), bg_fft)
        
    bg_fft_magnitude = np.abs( bg_fft )
    obj_fft_magnitude = np.abs( obj_fft )

    # We use '-' sign to inverse optimixation problem from MAX searching to MIN searching
    map = -ifft( np.divide( np.divide( map, bg_fft_magnitude), obj_fft_magnitude ) )
        
    # We probably could normalize only by BG part, since obj part is constant... (?)
    #map = -ifft( np.divide( map, bg_fft_magnitude**2 ) )
        
    # if it is color image, we sum along channel dim too.
    if len(map.shape) > 2:
        map = np.sum( map, axis=-1 )

    # regular map (with normalized cross-correlations) w/o cyclic shifts.
    map = map[ : -obj_img_shape[0], : -obj_img_shape[1] ]

    return map

# https://courses.cs.washington.edu/courses/cse576/05sp/papers/MSR-TR-2004-92.pdf, p/19
def __build_fourier_ssd_map(self, bg_img, obj_img):
    transparency_mask = get_transparency_mask(obj_img)
    obj_img = filter_by_alpha_channel(obj_img)
    bg_img = filter_by_alpha_channel(bg_img)
    obj_img = obj_img * transparency_mask

    obj_img_shape = obj_img.shape

    augmented_obj_img = np.zeros( bg_img.shape )
    augmented_obj_img[ : obj_img.shape[0],  : obj_img.shape[1] ] = obj_img
    obj_img = augmented_obj_img
        
    bg_fft = fft( bg_img )  
    obj_fft = fft( obj_img )

    # TODO: add delta-function
    map = -ifft( np.sum( bg_img**2 + obj_img**2 ) - 2 * np.multiply( np.conjugate(obj_fft), bg_fft) )
        
    # if it is color image, we sum along channel dim too.
    if len(map.shape) > 2:
        map = np.sum( map, axis=-1 )

    # regular map (with normalized cross-correlations) w/o cyclic shifts.
    map = map[ : -obj_img_shape[0], : -obj_img_shape[1] ]

    return map

def __build_fourier_neg_correlation_map_with_alpha(self, bg_img, obj_img):
    transparency_mask = get_transparency_mask(obj_img)
    obj_img = filter_by_alpha_channel(obj_img)
    bg_img = filter_by_alpha_channel(bg_img)
    obj_img = obj_img * transparency_mask

    obj_img_shape = obj_img.shape

    augmented_obj_img = np.zeros( bg_img.shape )
    augmented_obj_img[ : obj_img.shape[0],  : obj_img.shape[1] ] = obj_img
    obj_img = augmented_obj_img

    augmented_transparency_mask = np.zeros( bg_img.shape )
    augmented_transparency_mask[ : transparency_mask.shape[0],  : transparency_mask.shape[1] ] = transparency_mask
    transparency_mask = augmented_transparency_mask
        
    bg_fft = fft( bg_img )  
    obj_fft = fft( obj_img )

    map = np.multiply( np.conjugate(bg_fft), obj_fft)
        
    #bg_fft_magnitude = np.abs(  np.multiply( np.conjugate( fft(transparency_mask) ), bg_fft) )
    bg_fft_magnitude = np.abs( np.multiply( np.conjugate( fft(transparency_mask) ), bg_fft) )
    #obj_fft_magnitude = np.abs( obj_fft )

    # We use '-' sign to inverse optimixation problem from MAX searching to MIN searching
    #map = -ifft( np.divide( np.divide( map, bg_fft_magnitude), obj_fft_magnitude ) )
        
    # We probably could normalize only by BG part, since obj part is constant... (?)
    #map = -ifft( np.divide( map, bg_fft_magnitude**2 ) )
    map = -ifft( np.divide( map, bg_fft_magnitude ) )
        
    # if it is color image, we sum along channel dim too.
    if len(map.shape) > 2:
        map = np.sum( map, axis=-1 )

    # regular map (with normalized cross-correlations) w/o cyclic shifts.
    map = map[ : -obj_img_shape[0], : -obj_img_shape[1] ]

    return map

# https://habr.com/ru/post/266129/
def __build_fourier_neg_correlation_map_dprotopopov(self, bg_img, obj_img):
    obj_img_shape = obj_img.shape

    E = np.zeros( bg_img.shape )
    E[ : obj_img.shape[0],  : obj_img.shape[1] ] = 1

    augmented_obj_img = np.zeros( bg_img.shape )
    augmented_obj_img[ : obj_img.shape[0],  : obj_img.shape[1] ] = obj_img
    obj_img = augmented_obj_img
        
    bg_fft = fft( bg_img )   # second 
    obj_fft = fft( obj_img ) # first
    E_fft = fft(E) # first 

    map = -ifft( np.multiply( np.conjugate(obj_fft), bg_fft) ) # data1        
    normalizer = ifft( np.multiply( np.conjugate( np.multiply( np.conjugate(E_fft), bg_fft) ), bg_fft)) # data2        
    map = np.divide( map+1, normalizer+1 )

    if len(map.shape) > 2:
        map = np.sum( map, axis=-1 )

    map = map[ : -obj_img_shape[0], : -obj_img_shape[1] ]

    return map


def __build_fourier_neg_correlation_map_dprotopopov_with_alpha(self, bg_img, obj_img):
    transparency_mask = get_transparency_mask(obj_img)
    obj_img = filter_by_alpha_channel(obj_img)
    bg_img = filter_by_alpha_channel(bg_img)
    obj_img = obj_img * transparency_mask

    obj_img_shape = obj_img.shape

    augmented_obj_img = np.zeros( bg_img.shape )
    augmented_obj_img[ : obj_img.shape[0],  : obj_img.shape[1] ] = obj_img
    obj_img = augmented_obj_img

    E = np.zeros( bg_img.shape )
    E[ : transparency_mask.shape[0],  : transparency_mask.shape[1] ] = transparency_mask
        
    bg_fft = fft( bg_img )   # second 
    obj_fft = fft( obj_img ) # first
    E_fft = fft(E) # first 

    map = -ifft( np.multiply( np.conjugate(obj_fft), bg_fft) ) # data1        
    normalizer = ifft( np.multiply( np.conjugate( np.multiply( np.conjugate(E_fft), bg_fft) ), bg_fft)) # data2        
    map = np.divide( map+1, normalizer+1 )

    if len(map.shape) > 2:
        map = np.sum( map, axis=-1 )

    map = map[ : -obj_img_shape[0], : -obj_img_shape[1] ]

    return map

