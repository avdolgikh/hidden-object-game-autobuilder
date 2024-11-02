import math
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.resnet50 import preprocess_input



def preprocess_for_resnet(images):
    img_data = []
    for img in images:
        img = preprocess_input(img)
        img_data.append( img )
    return np.array(img_data)

def get_resnet_2c():
    model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling=None)    
    output = model.get_layer("conv2_block3_out").output
    return keras.models.Model( inputs = model.input, outputs = output )

def get_hog_similarity_resnet(av_pool_size, av_pool_stride):
    bg_model = keras.Sequential()
    bg_model.add( get_resnet_2c() )
    bg_model.add( keras.layers.AveragePooling2D(pool_size=av_pool_size, strides=av_pool_stride, padding='valid') )

    obj_model = keras.Sequential()
    obj_model.add( get_resnet_2c() )
    obj_model.add( keras.layers.GlobalAveragePooling2D() )

    output = keras.layers.Dot(axes=-1, normalize=True)( [ bg_model.output, obj_model.output ] )
    
    model = keras.models.Model( inputs = [bg_model.input, obj_model.input], outputs = output )        
    #model.summary()
    return model


if __name__ == '__main__':
    
    print("================== Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    #model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling=None) 
    #model.summary()

    # BG:
    path = r'..\..\data\third_stage_bgs\flower garden\bg_1.jpg'
    bg_img = cv2.imread( path, cv2.IMREAD_UNCHANGED )
    print("bg_img.shape:",  bg_img.shape)

    bg_img_data = preprocess_for_resnet( [ bg_img ] )
    print( "bg_img_data.shape:", bg_img_data.shape )

    # OBJ:
    path = r'..\..\data\third_stage_objs\1\H_1_Giant.png'
    obj_img = cv2.imread( path, cv2.IMREAD_UNCHANGED )
    print("obj_img.shape:",  obj_img.shape)

    obj_img_data = preprocess_for_resnet( [ obj_img ] )
    print( "obj_img_data.shape:", obj_img_data.shape )

    # ================
    
    av_pool_size = ( int(round(obj_img.shape[0] / 4)), int(round(obj_img.shape[1] / 4)) )
    print("av_poool_size:",  av_pool_size)
    
    av_pool_stride = 1 # int( np.max( [ 1, int(math.floor(bg_img.shape[0] / 240)) ] ) )
    print("stride:",  av_pool_stride)

    # -------------------- 

    model = get_hog_similarity_resnet(av_pool_size, av_pool_stride)   

    embeddings = model.predict( [ bg_img_data[:, :, :, :3], obj_img_data[:, :, :, :3] ] )
    print( "embeddings.shape:", embeddings.shape )

    


    