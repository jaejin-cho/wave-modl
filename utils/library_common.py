##########################################################
# %%
# Library for tensorflow 2.2.0
##########################################################

import sys, os
import numpy as np

from matplotlib import pyplot as plt

# tensorflow
import tensorflow as tf

# keras
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Activation, BatchNormalization, \
                                Conv2DTranspose, LeakyReLU, concatenate, Lambda, Conv3D, Add
from tensorflow.keras.models import Model    

##########################################################
# %%
# define common functions
##########################################################

def mosaic(img, num_row, num_col, fig_num, clim, title = '', use_transpose = False, use_flipud = False):
    
    fig = plt.figure(fig_num)
    fig.patch.set_facecolor('black')

    if img.ndim < 3:
        img_res = img
        plt.imshow(img_res)
        plt.gray()        
        plt.clim(clim)
    else:        
        if img.shape[2] != (num_row * num_col):
            print('sizes do not match')    
        else:               
            if use_transpose:
                for slc in range(0, img.shape[2]):
                    img[:,:,slc] = np.transpose(img[:,:,slc])
            
            if use_flipud:
                img = np.flipud(img)                
            
            img_res = np.zeros((img.shape[0]*num_row, img.shape[1]*num_col))            
            idx = 0
            
            for r in range(0, num_row):
                for c in range(0, num_col):
                    img_res[r*img.shape[0] : (r+1)*img.shape[0], c*img.shape[1] : (c+1)*img.shape[1]] = img[:,:,idx]
                    idx = idx + 1
        plt.imshow(img_res)
        plt.gray()        
        plt.clim(clim)
        
    plt.suptitle(title, color='white', fontsize=20)   
    

##########################################################
# %%
# tensorflow functions
##########################################################

def np_c2r(x):
    return np.stack((np.real(x),np.imag(x)),axis=-1)


def np_r2c(x):
    return x[...,0]+1j*x[...,1]


def np_c2r5(x):    
    res = np.stack(  ( np.real(x[...,0]),np.imag(x[...,0]), \
                       np.real(x[...,1]),np.imag(x[...,1]), \
                       np.real(x[...,2]),np.imag(x[...,2]), \
                       np.real(x[...,3]),np.imag(x[...,3]), \
                       np.real(x[...,4]),np.imag(x[...,4])  ),   axis = -1  )    
    return res

def np_r2c5(x):          
    res = np.stack( (  x[...,0]+1j*x[...,1],\
                       x[...,2]+1j*x[...,3],\
                       x[...,4]+1j*x[...,5],\
                       x[...,6]+1j*x[...,7],\
                       x[...,8]+1j*x[...,9] ),    axis = -1  )
    return res


c2r=Lambda(lambda x:tf.stack([tf.math.real(x),tf.math.imag(x)],axis=-1))
r2c=Lambda(lambda x:tf.complex(x[...,0],x[...,1]))


c2r5=Lambda(lambda x:tf.stack([tf.math.real(x[...,0]),tf.math.imag(x[...,0]),\
                               tf.math.real(x[...,1]),tf.math.imag(x[...,1]),\
                               tf.math.real(x[...,2]),tf.math.imag(x[...,2]),\
                               tf.math.real(x[...,3]),tf.math.imag(x[...,3]),\
                               tf.math.real(x[...,4]),tf.math.imag(x[...,4])],axis=-1))

r2c5=Lambda(lambda x:tf.stack([tf.complex(x[...,0],x[...,1]),\
                               tf.complex(x[...,2],x[...,3]),\
                               tf.complex(x[...,4],x[...,5]),\
                               tf.complex(x[...,6],x[...,7]),\
                               tf.complex(x[...,8],x[...,9])], axis=-1))
    
def nrmse_loss(y_true, y_pred):
    # return 100 * (K.sqrt(K.sum(K.square(y_pred - y_true)))) / (K.sqrt(K.sum(K.square(y_true))))
    return 100 * (K.sqrt(K.sum(K.square(y_pred - y_true)))) / (K.sqrt(K.sum(K.square(y_true))))


class tfft3(Layer):
    def __init__(self, **kwargs):
        super(tfft3, self).__init__(**kwargs)

    def build(self, input_shape):
        super(tfft3, self).build(input_shape)

    def call(self, x):
        xc = r2c5(x[0])
        
        # fft3
        t0 = tf.transpose(xc,[0,4,1,2,3])
        t0 = tf.signal.ifftshift(t0, axes=(2,3,4)) 
        t1 = tf.signal.fft3d(t0)
        t1 = tf.signal.fftshift(t1, axes=(2,3,4)) 
        t2 = tf.transpose(t1,[0,2,3,4,1])
                
        return c2r5(t2)


class tifft3(Layer):
    def __init__(self, **kwargs):
        super(tifft3, self).__init__(**kwargs)

    def build(self, input_shape):
        super(tifft3, self).build(input_shape)

    def call(self, x):
        xc = r2c5(x[0])
        
        # ifft3
        t0 = tf.transpose(xc,[0,4,1,2,3])
        t0 = tf.signal.ifftshift(t0, axes=(2,3,4)) 
        t1 = tf.signal.ifft3d(t0)
        t1 = tf.signal.fftshift(t1, axes=(2,3,4)) 
        t2 = tf.transpose(t1,[0,2,3,4,1])
        
        return c2r5(t2)

##########################################################
# %%
# U-net functions
##########################################################

def conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type='relu', USE_BN=True, layer_name=''):
    with K.name_scope(layer_name):
        x = Conv2D(num_out_chan, kernel_size, activation=None, padding='same', kernel_initializer='truncated_normal')(x)
        if USE_BN:
            x = BatchNormalization()(x)            
        if activation_type == 'LeakyReLU':
            return LeakyReLU()(x)
        else:
            return Activation(activation_type)(x)
        
def conv2Dt_bn_nonlinear(x, num_out_chan, kernel_size, activation_type='relu', USE_BN=True, layer_name=''):
    with K.name_scope(layer_name):
        x = Conv2DTranspose(num_out_chan, kernel_size, strides=(2, 2), padding='same', kernel_initializer='truncated_normal')(x)
        if USE_BN:
            x = BatchNormalization()(x)
            
        if activation_type == 'LeakyReLU':
            return LeakyReLU()(x)
        else:
            return Activation(activation_type)(x)

def conv2D_bn_softmax(x, num_out_chan, kernel_size, USE_BN = True, layer_name=''):
    with K.name_scope(layer_name):
        x = Conv2D(num_out_chan, kernel_size, activation=None, padding='same', kernel_initializer='truncated_normal')(x)
        if USE_BN:
            x = BatchNormalization()(x)               
        return Activation('softmax')(x)
    

# Conv3D -> Batch Norm -> Nonlinearity
def conv3D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type='relu', USE_BN=True, layer_name=''):
    with K.name_scope(layer_name):
        x = Conv3D(num_out_chan, kernel_size, activation=None, padding='same', kernel_initializer='truncated_normal')(x)
        if USE_BN:
            x = BatchNormalization()(x)
        if activation_type == 'LeakyReLU':
            return LeakyReLU()(x)
        else:
            return Activation(activation_type)(x)


def RegConvLayers(nx,ny,nz,ne,nLayers,num_filters):

    input_x     = Input(shape=(nx,ny,nz,2*ne), dtype = tf.float32)
    filter_size = (3,3,2)
    bool_USE_BN = True
    AT          = 'LeakyReLU'

    rg_term     = input_x
    for lyr in range(0,nLayers):
        rg_term = conv3D_bn_nonlinear(rg_term, num_filters, filter_size, activation_type=AT, USE_BN=bool_USE_BN, layer_name='')

    # go to image space
    rg_term = conv3D_bn_nonlinear(rg_term, 2*ne, (1,1,1), activation_type='linear', USE_BN=False, layer_name='')

    # skip connection
    rg_term = Add()([rg_term,input_x])

    return Model(inputs     =   input_x, outputs    =   rg_term)