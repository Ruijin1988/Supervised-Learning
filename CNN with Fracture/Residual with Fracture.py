'''Residual block by Keunwoo Choi (keunwoo.choi@qmul.ac.uk)

It is based on "Deep Residual Learning for Image Recognition" (http://arxiv.org/abs/1512.03385)
and "Identity Mappings in Deep Residual Networks" (http://arxiv.org/abs/1603.05027).
'''
import keras
# from keras.models import Sequential, Graph
from keras.layers.core import Layer, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge
from keras.models import Model
import pdb

# W1=sio.loadmat('W1_temp')['W1_temp']
# W2=sio.loadmat('W2_temp')['W2_temp']
# W3=sio.loadmat('W3_temp')['W3_temp']
# W1=np.asarray(W1, dtype=np.float32)
# W2=np.asarray(W2, dtype=np.float32)
# W3=np.asarray(W3, dtype=np.float32)

def building_residual_block(input_shape, n_feature_maps, n_feat_next, image_patch_sizes,conv_idx,
                            kernel_sizes=None, kernel2_sizes=None, n_skip=2, is_subsample=False,
                            subsample=None):
    '''
    [1] Building block of layers for residual learning.
        Code based on https://github.com/ndronen/modeling/blob/master/modeling/residual.py
        , but modification of (perhaps) incorrect relu(f)+x thing and it's for conv layer

    [2] ----This comment used to be valid. Now it is not, but I failed to track since when.----
        ----Now the comment below is incorrect, I am using strided convolution here.----
        ----invalid comment------------------------------------------------------------------------------
        | MaxPooling is used instead of strided convolution to make it easier                           |
        | to set size(output of short-cut) == size(output of conv-layers).                              |
        | If you want to remove MaxPooling,                                                             |
        |    i) change (border_mode in Convolution2D in shortcut), 'same'-->'valid'                     |
        |    ii) uncomment ZeroPadding2D in conv layers.                                                |
        |        (Then the following Conv2D is not the first layer of this container anymore,           |
        |         so you can remove the input_shape in the line 101, the line with comment #'OPTION' )  |
        -------------------------------------------------------------------------------------------------

    [3] It can be used for both cases whether it subsamples or not.

    [4] In the short-cut connection, I used 1x1 convolution to increase #channel.
        It occurs when is_expand_channels == True

    input_shape = (None, num_channel, height, width)
    n_feature_maps: number of feature maps. In ResidualNet it increases whenever image is downsampled.
    kernel_sizes : list or tuple, (3,3) or [3,3] for example
    n_skip       : number of layers to skip
    is_subsample : If it is True, the layers subsamples by *subsample* to reduce the size.
    subsample    : tuple, (2,2) or (1,2) for example. Used only if is_subsample==True
    '''
    # is_expand_channels == True when num_channels increases.
    #    E.g. the very first residual block (e.g. 1->64, 3->128, 128->256, ...)

    import scipy.io as sio
    #from keras.utils.theano_utils import sharedX
    W1=sio.loadmat('W1_temp')['W1_temp']
    W2=sio.loadmat('W2_temp')['W2_temp']
    W3=sio.loadmat('W3_temp')['W3_temp']
    W1=np.asarray(W1, dtype=np.float32)
    W2=np.asarray(W2, dtype=np.float32)
    W3=np.asarray(W3, dtype=np.float32)


    kernel_row, kernel_col = kernel_sizes
    kernel_row2, kernel_col2 = kernel2_sizes

#     if n_feature_maps!=n_feat_next:
    print('conv_idx=',conv_idx)
    kernel_sizes_pre=image_patch_sizes[conv_idx-1]
    kernel_row_pre, kernel_col_pre = kernel_sizes_pre

    # ***** VERBOSE_PART *****
    print ('   - New residual block with')
    print ('      input shape:', input_shape)
    print ('      kernel size:', kernel_sizes)

    is_expand_channels = not (input_shape[0] == n_feature_maps)
    if is_expand_channels:
        print ('      - Input channels: %d ---> num feature maps on out: %d' % (input_shape[0], n_feature_maps)  )
    if is_subsample:
        print ('      - with subsample:', subsample)
    # set input
    x = Input(shape=(input_shape))
    # ***** SHORTCUT PATH *****
    if is_subsample: # subsample (+ channel expansion if needed)
        shortcut_y = Convolution2D(n_feature_maps, kernel_sizes[0], kernel_sizes[1],
                                    subsample=subsample,
                                    border_mode='same')(x)
        print ('short cut kernel sizes=',n_feature_maps, kernel_sizes[0], kernel_sizes[1])
    else: # channel expansion only (e.g. the very first layer of the whole networks)
        if is_expand_channels:
            shortcut_y = Convolution2D(n_feature_maps, 1, 1, border_mode='same')(x)
        else:
            # if no subsample and no channel expension, there's nothing to add on the shortcut.
            shortcut_y = x
        print ('short cut kernel sizes=',n_feature_maps, kernel_sizes[0], kernel_sizes[1])
    # ***** CONVOLUTION_PATH *****
    conv_y = x
    for i in range(n_skip):
        conv_y = BatchNormalization(axis=1, mode=2)(conv_y)
        conv_y = Activation('relu')(conv_y)
        if i==0 and is_subsample: # [Subsample at layer 0 if needed]
            conv_y = Convolution2D(n_feature_maps, kernel_row_pre, kernel_col_pre,
                                    subsample=subsample,
                                    border_mode='same')(conv_y)
            print ('kernel sizes 1=',conv_idx,n_feature_maps, kernel_row_pre, kernel_col_pre)
        else:
            conv_y = Convolution2D(n_feature_maps, kernel_row, kernel_col, border_mode='same')(conv_y)
            print ('kernel sizes 2=',conv_idx,n_feature_maps, kernel_row, kernel_col)
    # output
    y = merge([shortcut_y, conv_y], mode='sum')
    block = Model(input=x, output=y)
    print ('        -- model was built.')
    print(block.layers[3].W.get_value().shape)
#     block.summary()

    if conv_idx==4:
        print('feature=40, W size=',block.layers[3].W.get_value().shape)
        print('feature=40, W2 size=',W2.shape)
        block.layers[3].W.set_value(W2)
#         block.layers[6].W.set_value(W2)
    elif conv_idx==9:
        print('feature=288, W size=',block.layers[3].W.get_value().shape)
        print('feature=288, W3 size=',W3.shape)
        block.layers[3].W.set_value(W3)
#         block.layers[6].W.set_value(W3)
    return block


from __future__ import print_function
import sys

sys.setrecursionlimit(99999)
import pdb

import numpy as np

np.random.seed(1337)  # for reproducibility

import keras

from keras.datasets import mnist, cifar10
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Convolution2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

batch_size = 10
nb_classes = 1
nb_epoch = 20


def compute_padding_length(length_before, stride, length_conv):
    ''' Assumption: you want the subsampled result has a length of floor(original_length/stride).
    '''
    N = length_before
    F = length_conv
    S = stride
    if S == F:
        return 0
    if S == 1:
        return (F - 1) / 2
    for P in range(S):
        if (N - F + 2 * P) / S + 1 == N / S:
            return P
    return 0


def design_for_residual_blocks(num_channel_input=1):
    ''''''
    model = Sequential()  # it's a CONTAINER, not MODEL
    # set numbers
    num_big_blocks = 17
    #     image_patch_sizes = [[3,3]]*(num_big_blocks+1)
    image_patch_sizes = [[6, 6], [6, 6], [6, 6], [6, 6], [9, 9], [9, 9], [9, 9], [9, 9], [9, 9], [9, 9], [9, 9], [9, 9],
                         [9, 9], [9, 9], [9, 9], [9, 9],
                         [9, 9], [9, 9], [9, 9], [9, 9], [9, 9], [9, 9], [9, 9], [9, 9], [9, 9], [9, 9], [9, 9], [9, 9],
                         [9, 9], [9, 9], [9, 9], [9, 9]]
    pool_sizes = [(2, 2)] * num_big_blocks
    #     n_features =      [24,24,24,24,40,40,40, 288,288, 288,512,512, 1024]
    #     n_features_next = [24,24,24,40,40,40,288,288,288, 512,512,1024,1024]
    n_features = [24, 24, 24, 24, 24, 40, 40, 40, 40, 40, 288, 288, 312, 312, 576, 576, 576]
    n_features_next = [24, 24, 24, 24, 40, 40, 40, 40, 40, 288, 288, 288, 288, 312, 312, 576, 576]
    height_input = 144
    width_input = 144

    for conv_idx in range(num_big_blocks):
        n_feat_here = n_features[conv_idx]
        n_feat_next = n_features_next[conv_idx]
        # residual block 0
        model.add(building_residual_block((num_channel_input, height_input, width_input),
                                          n_feat_here, n_feat_next, image_patch_sizes, conv_idx,
                                          kernel_sizes=image_patch_sizes[conv_idx],
                                          kernel2_sizes=image_patch_sizes[conv_idx + 1]
                                          ))

        # residual block 1 (you can add it as you want (and your resources allow..))
        if False:
            model.add(building_residual_block((n_feat_here, height_input, width_input),
                                              n_feat_here,
                                              kernel_sizes=image_patch_sizes[conv_idx],
                                              kernel2_sizes=image_patch_sizes[conv_idx + 1]
                                              ))

        # the last residual block N-1
        # the last one : pad zeros, subsamples, and increase #channels
        pad_height = compute_padding_length(height_input, pool_sizes[conv_idx][0], image_patch_sizes[conv_idx][0])
        pad_width = compute_padding_length(width_input, pool_sizes[conv_idx][1], image_patch_sizes[conv_idx][1])
        model.add(ZeroPadding2D(padding=(pad_height, pad_width)))
        height_input += 2 * pad_height
        width_input += 2 * pad_width
        n_feat_next = n_features_next[conv_idx]
        if n_features[conv_idx] != n_features_next[conv_idx]:
            model.add(building_residual_block((n_feat_here, height_input, width_input),
                                              n_feat_next, n_feat_next, image_patch_sizes, conv_idx,
                                              kernel_sizes=image_patch_sizes[conv_idx],
                                              kernel2_sizes=image_patch_sizes[conv_idx + 1],
                                              is_subsample=True,
                                              subsample=pool_sizes[conv_idx]
                                              ))

        height_input, width_input = model.output_shape[2:]
        # width_input  = int(width_input/pool_sizes[conv_idx][1])
        num_channel_input = n_feat_next

    # Add average pooling at the end:
    print('Average pooling, from (%d,%d) to (1,1)' % (height_input, width_input))
    model.add(AveragePooling2D(pool_size=(height_input, width_input)))
    return model


def get_residual_model(is_mnist=True, img_channels=1, img_rows=28, img_cols=28):
    model = keras.models.Sequential()
    first_layer_channel = 24

    W1 = sio.loadmat('W1_temp')['W1_temp']
    W1 = np.asarray(W1, dtype=np.float32)

    if is_mnist:  # size to be changed to 32,32
        model.add(ZeroPadding2D((2, 2), input_shape=(img_channels, img_rows, img_cols)))  # resize (28,28)-->(32,32)
        # the first conv
        model.add(Convolution2D(first_layer_channel, 3, 3, border_mode='same'))
    else:
        model.add(Convolution2D(first_layer_channel, 3, 3, border_mode='same',
                                input_shape=(img_channels, img_rows, img_cols)))

    model.add(Activation('relu'))
    # [residual-based Conv layers]
    residual_blocks = design_for_residual_blocks(num_channel_input=first_layer_channel)
    #     residual_blocks.layers[2].W.get_value().shape
    model.add(residual_blocks)
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # [Classifier]
    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('linear'))
    # [END]
    #     print(model.layers[1].W.get_value().shape)
    model.summary()
    print('feature=24, W size=', model.layers[1].W.get_value().shape)
    return model


if __name__ == '__main__':

    is_mnist = True
    is_cifar10 = not is_mnist

    if is_mnist:
        import scipy.io as sio
        import numpy as np

        img_rows, img_cols = 144, 144
        img_channels = 1
        WB = sio.loadmat('WB_small.mat')['WB_small']
        Y_data = sio.loadmat('true.mat')['skt']
        Y_data = (Y_data - min(Y_data)) / (max(Y_data) - min(Y_data))
        X_data = np.reshape(WB, (100, 1, img_rows, img_cols))
        X_train = X_data[0:80];
        X_test = X_data[80:100]
        Y_train = Y_data[0:80];
        Y_test = Y_data[80:100]
        print(' == MNIST ==')
    else:
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        img_rows, img_cols = 32, 32
        img_channels = 3
        print(' == CIFAR10 ==')

    import scipy.io as sio

    # from keras.utils.theano_utils import sharedX
    W1 = sio.loadmat('W1_temp')['W1_temp']
    W1 = np.asarray(W1, dtype=np.float32)

    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255
    # X_train = (X_train - np.mean(X_train))/np.std(X_train)
    # X_test = (X_test - np.mean(X_test))/np.std(X_test)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    model = get_residual_model(is_mnist=is_mnist, img_channels=img_channels, img_rows=img_rows, img_cols=img_cols)
    model.layers[1].W.set_value(W1)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
