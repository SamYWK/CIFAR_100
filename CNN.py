# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 22:09:46 2018

@author: pig84
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

print("Tensorflow Version {}".format(tf.VERSION))

def read_data(filename):
    '''
    READ DATA & Preprocessing
    '''
    #read data
    df = pd.read_pickle(filename)
    #split data to features label
    label = np.array(df['fine_labels']).reshape(-1, 1)
    features = np.array(df['data'])
    features = (features/255.0).reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])
    enc = OneHotEncoder()
    label = enc.fit_transform(label).toarray()
    return train_test_split(features, label, test_size = 0.2, random_state = 1)

def main():
    X_train, X_test, y_train, y_test = read_data('./cifar-100-python/train')
    n = X_train.shape[0]
    n_test = X_test.shape[0]
    
    batch_size = 200
    epochs = 50
    learning_rate = 1e-4
    
    g_1 = tf.Graph()
    with g_1.as_default():
        with tf.name_scope('X_placeholder'):
            X_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
        with tf.name_scope('y_placeholder'):
            y_placeholder = tf.placeholder(tf.float32, [None, 100])
        with tf.name_scope('keep_prob_1'):
            keep_prob_1 = tf.placeholder(tf.float32)
        with tf.name_scope('keep_prob_2'):
            keep_prob_2 = tf.placeholder(tf.float32)
            
        #L2 regularizer
        regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-4)
            
        x1 = tf.layers.conv2d(
                inputs = X_placeholder,
                filters = 32,
                kernel_size = 3,
                padding = 'same',
                activation = tf.nn.relu,
                kernel_regularizer = regularizer)
        x1 = tf.layers.batch_normalization(x1)
        
        x2 = tf.layers.conv2d(
                inputs = x1,
                filters = 32,
                kernel_size = 3,
                padding = 'same',
                activation = tf.nn.relu,
                kernel_regularizer = regularizer)
        x2 = tf.layers.batch_normalization(x2)
        pool2 = tf.nn.dropout(tf.layers.max_pooling2d(inputs = x2, pool_size = [2, 2], strides = 2), keep_prob_1)
        
        x3 = tf.layers.conv2d(
                inputs = pool2,
                filters = 64,
                kernel_size = 3,
                padding = 'same',
                activation = tf.nn.relu,
                kernel_regularizer = regularizer)
        x3 = tf.layers.batch_normalization(x3)

        x4 = tf.layers.conv2d(
                inputs = x3,
                filters = 64,
                kernel_size = 3,
                padding = 'same',
                activation = tf.nn.relu,
                kernel_regularizer = regularizer)
        x4 = tf.layers.batch_normalization(x4)
        pool4 = tf.nn.dropout(tf.layers.max_pooling2d(inputs = x4, pool_size=[2, 2], strides=2), keep_prob_2)
        pool4_flat = tf.reshape(pool4, [-1, 8*8*64])
        x5 = tf.layers.dense(pool4_flat, 100, activation = None)
        
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = x5, labels = y_placeholder)) + tf.losses.get_regularization_loss()
        with tf.name_scope('train_step'):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            
        #prediction
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(x5, 1), tf.argmax(y_placeholder, 1))
            correct_count = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_count)
            
        #initializer
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = True)
        config.gpu_options.allow_growth = True

        #saver
        saver = tf.train.Saver()
        with tf.Session(config = config) as sess:
            numbers = np.array([])
            sess.run(init)
            #saver.restore(sess, "./CNN_CIFAR10_8layers/model.ckpt")
#            start_time = time.time()
            for epoch in range(epochs):
                for batch in range(int (n / batch_size)):
                    batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                    batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                    sess.run(train_step, feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys, keep_prob_1: 0.8, keep_prob_2: 0.7})

                    if batch % 500 == 0:
                        
                        score = 0
                        for test_batch in range(int (n_test / batch_size)):
                            batch_xs_test = X_test[(test_batch*batch_size) : (test_batch+1)*batch_size]
                            batch_ys_test = y_test[(test_batch*batch_size) : (test_batch+1)*batch_size]
                            score += sum(sess.run(correct_count, feed_dict = {X_placeholder : batch_xs_test, y_placeholder : batch_ys_test, keep_prob_1: 1.0, keep_prob_2: 1.0}))
                        for i in range(int (n_test / batch_size)*batch_size, n_test):
                            batch_xs_test = X_test[i].reshape(1, -1)
                            batch_ys_test = y_test[i].reshape(1, -1)
                            score += sum(sess.run(correct_count, feed_dict = {X_placeholder : batch_xs_test, y_placeholder : batch_ys_test, keep_prob_1: 1.0, keep_prob_2: 1.0}))

                        print('Epochs :', epoch, 'Accuracy :', (score/n_test))
#                        numbers = np.append(numbers, score/n_test)
                        
                #saver.save(sess, "./CNN_CIFAR10_8layers/model.ckpt")
if __name__ == '__main__':
    main()