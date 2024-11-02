import numpy as np
import tensorflow
import keras
import keras.backend as K
from keras.utils import plot_model
import json
import os

    
def add_dense_layers(layer, n_hidden_units, dropout_rates, batch_normalization, output_layer_name=None):
    for i in range(len(n_hidden_units)):
        if dropout_rates is not None and dropout_rates[i] > 0:
            layer = keras.layers.Dropout(dropout_rates[i])(layer)
        layer = keras.layers.Dense(int(n_hidden_units[i]), activation=None, use_bias=True)(layer)
        if batch_normalization:
            layer = keras.layers.BatchNormalization(axis=-1)(layer)

        if i < len(n_hidden_units)-1:
            name = None
            activation = 'relu'
        else:
            name = output_layer_name
            #activation = 'linear'
            activation = 'tanh'
            #activation = 'relu' # such behavior is in LIVE now! And it is right: otherwise there is NO unlinearity!
        
        layer = keras.layers.Activation(activation, name=name)(layer)
        print("Activation of '{}' is '{}'.".format(name, activation))
        
    if dropout_rates is not None and len(dropout_rates) > len(n_hidden_units) and dropout_rates[len(n_hidden_units)] > 0:
        layer = keras.layers.Dropout(dropout_rates[len(n_hidden_units)])(layer)
    return layer

def add_recurrent_layers(layer, mask_value, n_recurrent_state, lstm_activation, lstm_recurrent_activation,
                        dropout_rate, bidirectional, n_lstm):
    layer = keras.layers.Masking(mask_value=mask_value)(layer)
    
    for i in range(n_lstm):
        return_sequences = i < (n_lstm - 1)
        lstm = keras.layers.LSTM(n_recurrent_state, activation=lstm_activation, recurrent_activation=lstm_recurrent_activation,
                                use_bias=True, dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=return_sequences)
        if bidirectional:
            layer = keras.layers.Bidirectional(lstm, merge_mode="concat")(layer)
        else:
            layer = lstm(layer)

        if return_sequences:
            layer = keras.layers.Dropout(0.5)(layer)

    return layer

def add_similarity_layers(layer_1, layer_2):
    return keras.layers.Dot(-1, normalize=True)([layer_1, layer_2]) # It's cosine similarity!

def get_padding(kernel_height):
    # ((top_pad, bottom_pad), (left_pad, right_pad)) 
    """
    if kernel_size == 2:
        padding=((0, 1), (0, 0))    
    if kernel_size == 3:
        padding=((1, 1), (0, 0))
    if kernel_size == 4:
        padding=((1, 2), (0, 0))
    """
    top_pad = int((kernel_height - 1) / 2)
    bottom_pad = int(kernel_height / 2)    
    return ((top_pad, bottom_pad), (0, 0))

def conv_block(layer, input_width, kernel_height, n_filters):
    padding = get_padding(kernel_height)    
    layer = keras.layers.ZeroPadding2D(padding=padding)(layer)
    stride = 1
    padding = "valid"
    layer = keras.layers.Conv2D(filters=n_filters,
                                kernel_size=(kernel_height, input_width),
                                strides=(stride, stride),
                                activation='relu',                                
                                padding=padding,
                                use_bias=True)(layer)
    # Rotation
    layer = keras.layers.Permute(dims=(1, 3, 2))(layer)
    return layer

def add_conv_layers(layer, input_width, kernel_heights, n_filters, maxpool_height=5, maxpool_stride_vertical=3):    
    layer = keras.layers.Reshape(target_shape=(-1, input_width, 1))(layer)
    convs = []
    for kernel_height in kernel_heights:
        convs.append(conv_block(layer, input_width, kernel_height=kernel_height, n_filters=n_filters))
    if len(kernel_heights) > 1:    
        layer = keras.layers.Concatenate(axis=2)(convs)
    else:
        layer = convs[0](layer)
    layer = keras.layers.MaxPooling2D(pool_size=(maxpool_height, 1), strides=(maxpool_stride_vertical, 1))(layer)
    layer = keras.layers.Reshape(target_shape=(-1, n_filters * len(kernel_heights)))(layer)
    return layer

def add_output_layers(layer, n_output_labels, batch_normalization):
    if n_output_labels > 2:
        activation = "softmax"
    else:    
        activation = "sigmoid"
        n_output_labels = 1

    layer = keras.layers.Dense(n_output_labels, activation=None, use_bias=True)(layer)
    if batch_normalization:
        layer = keras.layers.BatchNormalization(axis=-1)(layer)
    layer = keras.layers.Activation(activation)(layer)
    return layer

def plot_model_structure(model, to_file=None):
    try:
        """
        Before that: in an active Conda environment:
        `pip install graphviz`
        `pip install pydot`
        `conda install graphviz`
        """
        if to_file is None:
            to_file = os.path.join('.', 'outputs', 'model.png')
        else:
            to_file = os.path.join('.', 'outputs', to_file)

        plot_model(model,
                   to_file=to_file,
                   show_shapes=True,
                   show_layer_names=True,
                   rankdir='LR')
    except Exception as ex:
        print(ex)

def get_optimizer(optimizer, learning_rate, momentum, decay):
    # https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    if optimizer == 'sgd':     
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=False)
    if optimizer == 'rmsprop':     
        optimizer = keras.optimizers.RMSprop(lr=learning_rate, decay=decay, rho=momentum) # rho ~ momentum?
    if optimizer == 'adam':     
        optimizer = keras.optimizers.Adam(lr=learning_rate, decay=decay)
    # Gradient Clipping: clipnorm=1. or clipvalue=0.5
    # TODO: look at min/max values of W of good and bad (AUC-ROC=0.5) NNs. => use min/max of good one for clipping.
    return optimizer

def get_metric(n_output_labels):
    if n_output_labels > 2:
        metric = keras.metrics.categorical_accuracy # categorical_accuracy # categorical_crossentropy
    else:
        metric = keras.metrics.binary_accuracy # binary_crossentropy
    return metric

def load_weights(model, weights_file):
    if weights_file is not None:
        try:
            model.load_weights(weights_file)
            print("================ MODEL WEIGHTS WERE LOADED! ========================")
        except Exception as ex:
            print(ex)




