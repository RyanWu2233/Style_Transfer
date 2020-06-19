# -*- coding: utf-8 -*-
"""
Module name: Style_Transfer
Author:      Ryan Wu
Date:        2020/04/17
Description: Fast style transfer
Training time: ?? seconds / epoch (@ RTX2080 ti GPU )
Refer to:    
    [1] "A Neural Algorithm of Artistic Style" by Leon.Gatys, 2015
    [2] "Image Style Transfer Using Convolutional Neural Networks" by Leon.Gatys. 2016
    
"""
#---------------------------------------------------- Import libraries
from __future__ import division, print_function, absolute_import, unicode_literals
import tensorflow as tf
#import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import time
#from scipy.optimize import fmin_l_bfgs_b

from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Reshape
#---------------------------------------------------- Style 
#S_path     = 'SHA00.jpg'                            # Colorful crayon painting
#S_path     = 'SHA01.jpg'                            # Rain princess 
#S_path     = 'SHA02.jpg'                            # Purple fantasy (Anime)
#S_path     = 'SHA03.jpg'                            # Flaming
#S_path     = 'SHA04.jpg'                            # Van gogh (Straay  night)
S_path     = 'SHA32.jpg'                            # 

#---------------------------------------------------- Hyper parameters
C_path     = 'CH02.jpg'                             # Content image path
S_weight   = 100                                    # Style loss weighting
C_weight   = 0.025                                  # Content loss weighting
TV_weight  = 1                                      # Total variation weighting

iterations = 11                                     # Optimization iterations
rpt_times  = 200                                    # Repeat times in one iteration
nrows      = 720                                    # Height of result image
S_layers   = ['block1_conv1', 'block2_conv1','block3_conv1', 'block4_conv1','block5_conv1']
C_layer    = ['block3_conv2']                       # Content layer
R_prefix   = 'R'                                    # Prefix for transferred result
 
#---------------------------------------------------- Pre-process (Do not modify) 
W , H      = load_img(C_path).size;                 # Width & height of content image
ncols      = int(nrows / H * W);                    # Width of result image        
T_layers   = S_layers + C_layer                     # All Feature layer
NSL        = len(S_layers);                         # Number of style layers
optimizer  = tf.keras.optimizers.Adam(lr=10000E-4) 
Rname      = R_prefix+ C_path[0:4] + '_'+S_path[0:5] 

#---------------------------------------------------- Utilities
def preprocess_image(path):                         #-- Load image and convert into VGG19 format
    img = load_img(path, target_size=(nrows, ncols)) # Load jpeg as image object
    img = img_to_array(img)                         # Convert object as numpy array
    img = np.expand_dims(img, axis = 0)             # Convert [H,W,C] -> [N,H,W,C]
    img = vgg19.preprocess_input(img)
    return img
    #...........................................
def deprocess_image(x):                             #-- Convert VGG19 into normal image array
    N, H, W, C = x.shape;                           # Load size infomation
    x = x.reshape((H,W,C))                          # Convert [N,H,W,C] -> [H,W,C]
    x[:, :, 0] += 103.939                           # Recover VGG19 offset
    x[:, :, 1] += 116.779                           #
    x[:, :, 2] += 123.68                            #
    x = x[:, :, ::-1]                               # BGR -> RGB
    x = np.clip(x, 0, 255).astype('uint8')          # Convert to UINT8
    return x 
    #...........................................
def gram_matrix(x):                                 #-- Compute gram matrix
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram
    #...........................................
def style_loss(style, result):                      #-- Compute Style loss
    S = gram_matrix(style)
    R = gram_matrix(result)
    size   = nrows * ncols * 3
    factor = 4 * (size**2)
    return K.sum(K.square(S-R)) / factor
    #...........................................
def content_loss(base, result):                     #-- Compute content loss
    return tf.reduce_sum(tf.square(result-base))
    #...........................................
def total_variation_loss(R):                        #-- Compute TV loss
    nrows = R.shape[1];  ncols = R.shape[2]
    a = K.square( R[:, :nrows-1, :ncols-1, :] - R[:, 1:, :ncols-1, :])
    b = K.square( R[:, :nrows-1, :ncols-1, :] - R[:, :nrows-1, 1:, :])
    return K.sum(K.pow(a+b,1.25)) 
    #...........................................
C_image  = preprocess_image(C_path);                # Content image
S_image  = preprocess_image(S_path);                # Style image      


#---------------------------------------------------- Neural network model
def get_vgg(layers):
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    model_outputs = [vgg.get_layer(name).output for name in layers]
    return Model(vgg.input, model_outputs)
    #...........................................
def transfer_net():    
    init= tf.constant_initializer(np.reshape(C_image,(nrows*ncols*3,))) # preload image    
    xin = Input(shape=(1))
    x1  = Dense(nrows*ncols*3, use_bias=False, 
                dtype='float32',
                kernel_initializer = init)(xin) 
    xout= Reshape((nrows,ncols,3), dtype='float32')(x1)    
    return Model(xin, xout)
    #...........................................
VGG      = get_vgg(T_layers);                       # Feature extractor
R_model  = transfer_net();                          # Result model
Gain     = np.ones(shape=(1,1))                     # Result intensity (1=original)


#---------------------------------------------------- Style transfer step
def show_result():
        gen = R_model(Gain)                         # Generated by transfer net
        res = deprocess_image(gen.numpy())          # Recover image format from VGG format
        plt.imshow(res); plt.show()                 # Show result on screen
        return res
    #...........................................
@tf.function    
def transfer_step():
    with tf.GradientTape() as tape:                 # Build gradient Tape
        R_image = R_model(Gain);        
        VGG_in  = tf.concat([C_image, S_image, R_image],axis= 0)
        VGG_F   = VGG(VGG_in)                       # Extract feature by VGG
        #--- ---
        TV_loss = total_variation_loss(R_image)        
        loss    = TV_weight * TV_loss
        C_loss  = content_loss(VGG_F[NSL][0,:,:,:], VGG_F[NSL][2,:,:,:])
        loss    = loss + C_weight * C_loss
        for SL in range(NSL):
            S_loss = style_loss(VGG_F[SL][1,:,:,:], VGG_F[SL][2,:,:,:])
            loss   = loss + S_weight / NSL * S_loss
        
    grads = tape.gradient(loss, R_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, R_model.trainable_variables))
    return loss

#---------------------------------------------------- Main loop    
def main():
    print('Begin transfer!')
    start= time.time();
    for iters in range(iterations):
        loss = transfer_step()
        if (iters) % 2 == 0:                        #-- Show result 
            img = show_result()
            fname = Rname + '_sim_%d.png' % iters
            save_img(fname, img)  
        if (iters) % 1 == 0:                        #-- Show data
          log_loss= np.log10(loss);
          duration= time.time()-start;
          start   = time.time();
          print('Iteration: {} ; loss= {}; duration = {} sec'.format(iters, log_loss, duration))
        for rpt in range(rpt_times): loss = transfer_step()
    show_result()
    print('Finished !')    
#----------------------------------------------------
main();  
#----------------------------------------------------
#----------------------------------------------------






