# https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33
# https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Network_Keras.ipynb
# https://keras.io/applications/#resnet
# https://medium.com/@franky07724_57962/using-keras-pre-trained-models-for-feature-extraction-in-image-clustering-a142c6cdf5b1
# https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c

import numpy as np 
import keras
from dl_modeling_utils import *
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2
import sklearn.metrics
import scipy

def preprocess_for_resnet(images):
    img_data = []
    for img in images:
        if img.shape[0] < 197:        
            img = cv2.resize(img, (197, 197))
        img = preprocess_input(img * 255.) # TODO: check that it is float32 [0, 255] => *255
        img_data.append( img )
    return np.array(img_data)

def read_image_for_resnet(img_path):
    # TODO: if float32 [0, 255] => *255
    #img = image.load_img(img_path, target_size=(224, 224))
    img = image.load_img(img_path, target_size=(197, 197))
    img_data = image.img_to_array(img)
    #img_data = np.expand_dims(img_data, axis=0)

    # https://stackoverflow.com/questions/47555829/preprocess-input-method-in-keras
    # Some models use images with values ranging from 0 to 1. Others from -1 to +1. Others use the "caffe" style, that is not normalized, but is centered.
    # From the source code (https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py#L72), Resnet is using the caffe style.
    img_data = preprocess_input(img_data)
    return img_data

def read_images_for_resnet(img_paths):
    img_data = []
    for img_path in img_paths:
        img_data.append( read_image_for_resnet(img_path) )    
    return np.array(img_data)

def get_resnet50_img_embeddings_model():
    # The default input size for this model is 224x224
    # pooling='avg' (for feature extraction when include_top is False)
    stage5_output = keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')
    # it is "activation_49" (after "bn5c_branch2c") followed by GAP, end of res5

    def build_model(layer_name):
        output = stage5_output.get_layer(layer_name).output
        output = keras.layers.GlobalAveragePooling2D()(output)
        # GlobalMaxPooling2D ???
        return keras.models.Model( inputs = stage5_output.input, outputs = output )
    
    #level3 = build_model("activation_30")   # !!! it is a mistake! it is b/w "bn4c_branch2b" and "res4c_branch2c"
        # Must be "activation_22" = end of res3 !!!!!
    #level2 = build_model("activation_20") # b/w bn3d_branch2a and res3d_branch2b
        # Must be "activation_10" = end of res2 !!!!!

    #stage1_output = build_model("max_pooling2d_1")

    stage2_output = build_model("activation_10")        # 2c

    #stage3_1_output = build_model("activation_13")     # 3a
    stage3_2_output = build_model("activation_16")      # 3b
    #stage3_3_output = build_model("activation_19")     # 3c
    stage3_output = build_model("activation_22")        # 3d

    #stage4_1_output = build_model("activation_25")     # 4a
    stage4_2_output = build_model("activation_28")      # 4b
    #stage4_3_output = build_model("activation_31")     # 4c
    stage4_4_output = build_model("activation_34")      # 4d
    #stage4_5_output = build_model("activation_37")     # 4e
    stage4_output = build_model("activation_40")        # 4f, end of res4 # = `stage4_output`

    #stage5_1_output = build_model("activation_43")     # 5a
    stage5_2_output = build_model("activation_46")      # 5b
    
    """
    Layers from original paper https://arxiv.org/pdf/1512.03385.pdf:
    conv1
    conv2_x
    conv3_x
    conv4_x
    conv5_x
    """


    #stage5_output.summary()
    #plot_model_structure(stage5_output)

    return stage2_output, stage3_2_output, stage3_output, stage4_2_output, stage4_4_output, stage4_output, stage5_2_output, stage5_output

def resnet_embeddings_similarity(u, v, metric="cosine"):
    #return sklearn.metrics.pairwise.cosine_similarity([u], [v])[0][0]
    if metric == "cosine":
        return 1 - scipy.spatial.distance.cosine(u, v)
    return 1. / (1. + scipy.spatial.distance.euclidean(u, v))


if __name__ == '__main__':
    
    model = get_resnet50_img_embeddings_model()
    
    img_paths = [
                    #r'..\..\data\bird.png',
                    #r'..\..\data\bird.png',
                    #r'..\..\data\candle.png'
                    r'..\..\data\hog images\daily hidden object\H_1_Banana (2).png',
                    r'..\..\data\hog images\daily hidden object\H_1_Bananas.png',
                    r'..\..\data\hog images\daily hidden object\H_1_Banana.png',
                    r'..\..\data\hog images\daily hidden object\H_1_Basketball (4).png',
                ]
    img_data = read_images_for_resnet(img_paths)    
    print( img_data.shape )
    
    embeddings = model.predict( img_data )
    #print( embeddings )
    print( embeddings.shape )
    
    print( resnet_embeddings_similarity( embeddings[0], embeddings[1]) )
    print( resnet_embeddings_similarity( embeddings[0], embeddings[2]) )
    print( resnet_embeddings_similarity( embeddings[1], embeddings[2]) )
    print( resnet_embeddings_similarity( embeddings[0], embeddings[3]) )


    