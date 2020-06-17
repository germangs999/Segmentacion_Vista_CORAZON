# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 23:23:14 2020

@author: germa
"""
from __future__ import division, print_function, absolute_import
import os
import numpy as np
import skimage.io as io
import cv2
import tensorflow as tf
from optparse import OptionParser
import sys
sys.path.append('./funcs/')
sys.path.append('./nets/')
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from util import *


def clasificacion_vista(vol):
    # # Hyperparams
    parser=OptionParser()
    parser.add_option("-d", "--dicomdir", dest="dicomdir", help = "dicomdir")
    parser.add_option("-g", "--gpu", dest="gpu", default = "0", help = "cuda device to use")
    parser.add_option("-m", "--model", dest="model")
    params, args = parser.parse_args()
    dicomdir = params.dicomdir
    model = params.model
    
    import vgg as network
    
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu        
    
    #Argumentos de la red
    #Nombres y carpetas npara leer el modelo y los volumenes DICOM
    model = "view_23_e5_class_11-Mar-2018"
    model_name = './models/' + model
    
    #Leer las clases (vistas) capaz de detectar el modelo
    with open("vistas.txt", "rb") as fp:   # Unpickling
        views = pickle.load(fp)
    
    feature_dim = 1
    label_dim = len(views)
    
    dim = (224, 224)
    vol_resized = np.zeros([vol.shape[0],224,224,1], dtype = 'uint8')
    for idx in range(vol.shape[0]):
        vol_resized[idx,:,:,0] = cv2.resize(vol[idx,:,:], dim, interpolation = cv2.INTER_AREA)
    
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    model = network.Network(0.0, 0.0, feature_dim, label_dim, False)
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, model_name)
    
    predictions=np.around(model.probabilities(sess, vol_resized), decimals = 3)
    prom_predict = np.mean(predictions,axis=0)
    etiqueta = views[np.where(prom_predict == np.amax(prom_predict))[0][0]]
    return etiqueta

class Unet(object):        
    def __init__(self, mean, weight_decay, learning_rate, label_dim, maxout = False):
        self.x_train = tf.compat.v1.placeholder(tf.float32, [None, 384, 384, 1])
        self.y_train = tf.compat.v1.placeholder(tf.float32, [None, 384, 384, label_dim])
        self.x_test = tf.compat.v1.placeholder(tf.float32, [None, 384, 384, 1])
        self.y_test = tf.compat.v1.placeholder(tf.float32, [None, 384, 384, label_dim])
        self.label_dim = label_dim
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.maxout = maxout

        self.output = self.unet(self.x_train, mean)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.output, labels = self.y_train))
        self.opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.pred = self.unet(self.x_test, mean, keep_prob = 1.0, reuse = True)
        self.loss_summary = tf.compat.v1.summary.scalar('loss', self.loss)
#         self.train_summary = tf.summary.scalar('training_accuracy', self.train_accuracy)
    
    # Gradient Descent on mini-batch
    def fit_batch(self, sess, x_train, y_train):
        _, loss, loss_summary = sess.run((self.opt, self.loss, self.loss_summary), feed_dict={self.x_train: x_train, self.y_train: y_train})
        return loss, loss_summary
    
    def predict(self, sess, x):
        prediction = sess.run((self.pred), feed_dict={self.x_test: x})
        return prediction

    def unet(self, input, mean, keep_prob = 0.5, reuse = None):
        width = 1
        weight_decay = 1e-12
        label_dim = self.label_dim
        with tf.compat.v1.variable_scope('vgg', reuse=reuse):
            input = input - mean
            pool_ = lambda x: max_pool(x, 2, 2)
            conv_ = lambda x, output_depth, name, padding = 'SAME', relu = True, filter_size = 3: conv(x, filter_size, output_depth, 1, weight_decay, name=name, padding=padding, relu=relu)
            deconv_ = lambda x, output_depth, name: deconv(x, 2, output_depth, 2, weight_decay, name=name)
            fc_ = lambda x, features, name, relu = True: fc(x, features, weight_decay, name, relu)
            
            conv_1_1 = conv_(input, int(64*width), 'conv1_1')
            conv_1_2 = conv_(conv_1_1, int(64*width), 'conv1_2')
            
            pool_1 = pool_(conv_1_2)

            conv_2_1 = conv_(pool_1, int(128*width), 'conv2_1')
            conv_2_2 = conv_(conv_2_1, int(128*width), 'conv2_2')
            
            pool_2 = pool_(conv_2_2)

            conv_3_1 = conv_(pool_2, int(256*width), 'conv3_1')
            conv_3_2 = conv_(conv_3_1, int(256*width), 'conv3_2')

            pool_3 = pool_(conv_3_2)

            conv_4_1 = conv_(pool_3, int(512*width), 'conv4_1')
            conv_4_2 = conv_(conv_4_1, int(512*width), 'conv4_2')

            pool_4 = pool_(conv_4_2)

            conv_5_1 = conv_(pool_4, int(1024*width), 'conv5_1')
            conv_5_2 = conv_(conv_5_1, int(1024*width), 'conv5_2')

            pool_5 = pool_(conv_5_2)

            conv_6_1 = tf.nn.dropout(conv_(pool_5, int(2048*width), 'conv6_1'), keep_prob)
            conv_6_2 = tf.nn.dropout(conv_(conv_6_1, int(2048*width), 'conv6_2'), keep_prob)
           
            up_7 = tf.concat([deconv_(conv_6_2, int(1024*width), 'up7'), conv_5_2], 3)
            
            conv_7_1 = conv_(up_7, int(1024*width), 'conv7_1')
            conv_7_2 = conv_(conv_7_1, int(1024*width), 'conv7_2')
            
            up_8 = tf.concat([deconv_(conv_7_2, int(512*width), 'up8'), conv_4_2], 3)
            
            conv_8_1 = conv_(up_8, int(512*width), 'conv8_1')
            conv_8_2 = conv_(conv_8_1, int(512*width), 'conv8_2')
            
            up_9 = tf.concat([deconv_(conv_8_2, int(256*width), 'up9'), conv_3_2], 3)
            
            conv_9_1 = conv_(up_9,int(256*width), 'conv9_1')
            conv_9_2 = conv_(conv_9_1, int(256*width), 'conv9_2')

            up_10 = tf.concat([deconv_(conv_9_2, int(128*width), 'up10'), conv_2_2], 3)
            
            conv_10_1 = conv_(up_10, int(128*width), 'conv10_1')
            conv_10_2 = conv_(conv_10_1, int(128*width), 'conv10_2')

            up_11 = tf.concat([deconv_(conv_10_2, int(64*width), 'up11'), conv_1_2], 3)
            
            conv_11_1 = conv_(up_11, int(64*width), 'conv11_1')
            conv_11_2 = conv_(conv_11_1, int(64*width), 'conv11_2')
            
            conv_12 = conv_(conv_11_2, label_dim, 'conv12_2', filter_size = 1, relu = False)
            return conv_12

def segmentacion_4c(vol):
    view = "a4c"
    mean = 24
    weight_decay = 1e-12
    learning_rate = 1e-4
    maxout = False
    sesses = []
    models = []
    if view == "a4c":
        g_1 = tf.Graph()
        with g_1.as_default():
            label_dim = 6 #a4c
            # label_dim = 8 #a4c
            sess1 = tf.compat.v1.Session()
            model1 = Unet(mean, weight_decay, learning_rate, label_dim , maxout = maxout)
            sess1.run(tf.compat.v1.local_variables_initializer())
            sess = sess1
            model = model1
        with g_1.as_default():
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess1,'./models/a4c_45_20_all_model.ckpt-9000')
        orig_images, images_rgb, images = volumenes_necesarios(vol)
        a4c_lv_segs, a4c_la_segs, a4c_lvo_segs, a4c_rv_segs, a4c_ra_segs, preds = extract_segs1(images, orig_images, model, sess, 2, 4, 1,3,5)
        #Arreglos en ceros
        a4c_lv_seg = np.zeros([384,384,len(a4c_lv_segs)])
        a4c_la_seg = np.zeros([384,384,len(a4c_lv_segs)])
        a4c_lvo_seg= np.zeros([384,384,len(a4c_lv_segs)])
        a4c_rv_seg = np.zeros([384,384,len(a4c_lv_segs)])
        a4c_ra_seg = np.zeros([384,384,len(a4c_lv_segs)])
        
        for idx in range(len(a4c_lv_segs)):
            a4c_lv_seg[:,:,idx] = a4c_lv_segs[idx][:]
            a4c_la_seg[:,:,idx] = a4c_la_segs[idx][:]
            a4c_lvo_seg[:,:,idx] = a4c_lvo_segs[idx][:]
            a4c_rv_seg[:,:,idx] = a4c_rv_segs[idx][:]
            a4c_ra_seg[:,:,idx] = a4c_ra_segs[idx][:]
        
        segmentacion = a4c_lv_seg + 2*a4c_la_seg + 3*a4c_lvo_seg + 4*a4c_rv_seg + 5*a4c_ra_seg
    else:
        print('Esa vista no está disponible')
        segmentacion = []
    
    return segmentacion

def volumenes_necesarios(ds):
    if len(ds[0].shape) == 2:
        volumen = np.zeros([ds.shape[0], ds.shape[1], ds.shape[2],3]).astype('uint8')
        volumen[:,:,:,0] = volumen[:,:,:,1] = volumen[:,:,:,2] = ds
        volumen_rec = np.zeros([volumen.shape[0],384,384,volumen.shape[3]]).astype('uint8') #Tamaño para unet en RGB
        volumen_rec_g = np.zeros([volumen.shape[0],384,384]).astype('uint8')
        for idx in range(volumen.shape[0]):
            volumen_rec[idx,:,:,:] = cv2.resize(volumen[idx,:,:,:], (384, 384))
            volumen_rec_g[idx,:,:] = cv2.resize(ds[idx,:,:], (384, 384))
        volumen_rec_g = volumen_rec_g[:,:,:,np.newaxis]
        
    elif len(ds[0].shape) == 3:
        if ds.shape[3] == 3:
            volumen = ds
            volumen_rec = np.zeros([volumen.shape[0],384,384,volumen.shape[3]]).astype('uint8') #Tamaño para unet en RGB
            volumen_rec_g = np.zeros([volumen.shape[0],384,384]).astype('uint8') #Tamaño para unet en gris
            for idx in range(volumen.shape[0]):
                volumen_rec[idx,:,:,:] = cv2.resize(volumen[idx,:,:,:], (384, 384))
                volumen_rec_g[idx,:,:] = cv2.cvtColor(volumen_rec[idx,:,:,:], cv2.COLOR_BGR2GRAY)
            volumen_rec_g = volumen_rec_g[:,:,:,np.newaxis]
        else:
            print('No se puede procesar ese tipo de volumen')
    else:
        print('No se puede procesar el volumen')
    return volumen, volumen_rec,volumen_rec_g
    

def create_seg(output, label):
    output = output.copy()
    output[output != label] = -1
    output[output == label] = 1
    output[output == -1] = 0
    return output

#a4c_lv_segs, a4c_la_segs, a4c_lvo_segs, preds = extract_segs(images, orig_images, model, sess, 2, 4, 1, 3,5),6,7
def extract_segs1(images, orig_images, model, sess, lv_label, la_label, lvo_label, rv_label, ra_label):
    segs = []
    preds = np.argmax(model.predict(sess, images[0:1])[0,:,:,:], 2)
    label_all = range(1, 8)
    label_good = [lv_label, la_label, lvo_label, rv_label, ra_label]
    for i in label_all:
        if not i in label_good:
            preds[preds == i] = 0
    for i in range(len(images)):
        seg = np.argmax(model.predict(sess, images[i:i+1])[0,:,:,:], 2)
        segs.append(seg)
    lv_segs = []
    lvo_segs = []
    la_segs = []
    rv_segs = []
    ra_segs =[]

    for seg in segs:
        la_seg = create_seg(seg, la_label)
        lvo_seg = create_seg(seg, lvo_label)
        lv_seg = create_seg(seg, lv_label)
        rv_seg = create_seg(seg, rv_label)
        ra_seg = create_seg(seg, ra_label)
        
        lv_segs.append(lv_seg)
        lvo_segs.append(lvo_seg)
        la_segs.append(la_seg)
        rv_segs.append(rv_seg)
        ra_segs.append(ra_seg)
        
    return lv_segs, la_segs, lvo_segs, rv_segs, ra_segs, preds

def segmentacion_a4c_vista(vol):
    vista = clasificacion_vista(vol)
    vol_segmentado = segmentacion_4c(vol)
    return vol_segmentado, vista
    