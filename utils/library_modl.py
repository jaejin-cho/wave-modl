##########################################################
# %%
# Import Library
##########################################################

import numpy as np
import tensorflow as tf

# keras
import tensorflow.keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Add
    
from tensorflow.keras.models import Model

# import custom functions 
from library_common import *

##########################################################
# %%
# tensorflow functions
##########################################################

class Aclass:
    def __init__(self, csm,mask,lam):
        with tf.name_scope('Ainit'):
            self.mask           =   mask
            self.csm            =   csm
            self.lam            =   lam

    def myAtA(self,img):
        with tf.name_scope('AtA'):
            coilImages  =   tf.expand_dims(self.csm,axis=-1)*tf.expand_dims(img,axis=0)
            kspace      =   tf.transpose(tf.signal.fft3d(tf.transpose(coilImages,[0,4,1,2,3])),[0,2,3,4,1])
            temp        =   kspace*self.mask
            coilImgs    =   tf.transpose(tf.signal.ifft3d(tf.transpose(temp,[0,4,1,2,3])),[0,2,3,4,1])
            coilComb    =   tf.reduce_sum(coilImgs*tf.math.conj(tf.expand_dims(self.csm,axis=-1)),axis=0)
            coilComb    =   coilComb+self.lam*img
        return coilComb


def myCG(A,rhs):

    rhs=r2c5(rhs)
    cond=lambda i,rTr,*_: tf.logical_and( tf.less(i,10), rTr>1e-5)
    def body(i,rTr,x,r,p):
        with tf.name_scope('cgBody'):
            Ap      =   A.myAtA(p)
            alpha   =   rTr / tf.cast(tf.reduce_sum(tf.math.conj(p)*Ap),dtype=tf.float32)
            alpha   =   tf.complex(alpha,0.)
            x       =   x + alpha * p
            r       =   r - alpha * Ap
            rTrNew  =   tf.cast( tf.reduce_sum(tf.math.conj(r)*r),dtype=tf.float32)
            beta    =   rTrNew / rTr
            beta    =   tf.complex(beta,0.)
            p       =   r + beta * p
        return i+1,rTrNew,x,r,p

    x       =   tf.zeros_like(rhs)
    i,r,p   =   0,rhs,rhs
    rTr     =   tf.cast( tf.reduce_sum(tf.math.conj(r)*r),dtype=tf.float32)
    loopVar =   i,rTr,x,r,p
    out     =   tf.while_loop(cond,body,loopVar,name='CGwhile',parallel_iterations=1)[2]
    return c2r5(out)

class myDC(Layer):

    def __init__(self, **kwargs):
        super(myDC, self).__init__(**kwargs)

        self.lam1 = self.add_weight(name='lam1', shape=(1,), initializer=tf.constant_initializer(value=0.015),
                                     dtype='float32', trainable=True)
        self.lam2 = self.add_weight(name='lam2', shape=(1,), initializer=tf.constant_initializer(value=0.015),
                                     dtype='float32', trainable=True)

    def build(self, input_shape):
        super(myDC, self).build(input_shape)

    def call(self, x):
        rhs, csm, mask = x
        lam3 = tf.complex(self.lam1 + self.lam2, 0.)

        def fn(tmp):
            c, m, r = tmp
            Aobj = Aclass(c, m, lam3)
            y = myCG(Aobj, r)
            return y

        inp = (csm, mask, rhs)
        # Mapping functions with multi-arity inputs and outputs
        rec = tf.map_fn(fn, inp, dtype=tf.float32, name='mapFn')
        return rec

    def lam_weight1(self, x):
        img = x[0]
        res = (self.lam1+self.lam2) * img
        return res

    def lam_weight2(self, x):
        in0, in1 = x
        res = self.lam1 * in0 + self.lam2 * in1
        return res


class rm_bg(Layer):
    def __init__(self, **kwargs):
        super(rm_bg, self).__init__(**kwargs)

    def build(self, input_shape):
        super(rm_bg, self).build(input_shape)

    def call(self, x):
        img, csm    = x
        rcsm        = tf.expand_dims(tf.reduce_sum(tf.math.abs(csm),axis=1), axis=-1)
        cmask       = tf.math.greater(rcsm,tf.constant(0,dtype=tf.float32))
        rec         = tf.cast(cmask,dtype=tf.float32) * img
        return rec


##########################################################
# %%
# network
##########################################################

def create_modl(nx, ny, rz, nc, ne, nLayers, num_block, num_filters = 64):

    # define the inputs
    input_c     = Input(shape=(nc,nx,ny,rz), dtype = tf.complex64,      name = 'input_c')
    input_m     = Input(shape=(nx,ny,rz,ne), dtype = tf.complex64,      name = 'input_m')
    input_Atb   = Input(shape=(nx,ny,rz,ne), dtype = tf.complex64,      name = 'input_a')

    dc_term     = c2r5(input_Atb)

    RegConv_k   = RegConvLayers(nx,ny,rz,ne,nLayers,num_filters)
    RegConv_i   = RegConvLayers(nx,ny,rz,ne,nLayers,num_filters)
    UpdateDC    = myDC()
    rmbg        = rm_bg()

    myFFT       = tfft3()
    myIFFT      = tifft3()

    for blk in range(0,num_block):
        # CNN Regularization
        rg_term_i   = RegConv_i(dc_term)
        rg_term_k   = myIFFT([RegConv_k(myFFT([dc_term]))])
        rg_term     = UpdateDC.lam_weight2([rg_term_i,rg_term_k])
        # AtA update
        rg_term     = Add()([c2r5(input_Atb), rg_term])

        # Update DC
        dc_term     = UpdateDC([rg_term,input_c,input_m])

    out_x = rmbg([dc_term,input_c])

    return Model(inputs     =   [ input_c, input_m, input_Atb],
                 outputs    =   out_x )



def create_sense(nx, ny, rz, nc, ne, nLayers, num_block, num_filters = 64):

    # define the inputs
    input_c     = Input(shape=(nc,nx,ny,rz), dtype = tf.complex64,      name = 'input_c')
    input_m     = Input(shape=(nx,ny,rz,ne), dtype = tf.complex64,      name = 'input_m')
    input_Atb   = Input(shape=(nx,ny,rz,ne), dtype = tf.complex64,      name = 'input_a')

    dc_term     = c2r5(input_Atb)

    UpdateDC    = myDC()
    rmbg        = rm_bg()

    for blk in range(0,num_block):
        # CNN Regularization
        
        rg_term     = UpdateDC.lam_weight1([dc_term])
        # AtA update
        rg_term     = Add()([c2r5(input_Atb), rg_term])

        # Update DC
        dc_term     = UpdateDC([rg_term,input_c,input_m])

    out_x = rmbg([dc_term,input_c])

    return Model(inputs     =   [ input_c, input_m, input_Atb],
                 outputs    =   out_x )


