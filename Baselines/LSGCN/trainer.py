#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from preprocess import gen_batch
from tester import model_inference
from base_model import build_model
from base_model import model_save

import tensorflow as tf
import numpy as np
import time


def model_train(inputs, blocks, args):
    '''
    Construct network structure, train and save model.
    :param inputs: instance of class Dataset, data source for training.
    :param blocks: list, channel configs of every layer.
    :param args: instance of class argparse, args for training.
    '''
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    C_i, batch_size, epoch = args.C_i, args.batch_size, args.epoch
    sum_path = args.sum_path

    x = tf.placeholder(tf.float32, [None, n_his + 1, n, C_i], name='data_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Model loss
    train_loss, pred = build_model(x, n_his, Ks, Kt, blocks, keep_prob)
    tf.summary.scalar('train_loss', train_loss)
    copy_loss = tf.add_n(tf.get_collection('copy_loss'))

    # Learning rate
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_len('train')
    if len_train % batch_size == 0:
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1
    lr = tf.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
    train_op = tf.train.RMSPropOptimizer(lr).minimize(train_loss)

    #  Run session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
        min_val = min_va_val = np.array([4e1, 1e5, 1e5] * len(step_idx))
        for i in range(epoch):
            start_time = time.time()
            for j, x_batch in enumerate(
                    gen_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):
                # Model forward
                sess.run(train_op, feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                # Show loss in every 50 steps
                if j % 50 == 0:
                    loss_value = \
                        sess.run([train_loss, copy_loss],
                                 feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                    print(f'Epoch {i:2d}, Step {j:3d}: [{loss_value[0]:.3f}, {loss_value[1]:.3f}]')
            print(f'Epoch {i:2d} Training Time {time.time() - start_time:.3f}s')

            start_time = time.time()
            min_va_val, min_val = \
                model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val)

            for ix in tmp_idx:
                va, te = min_va_val[ix - 2:ix + 1], min_val[ix - 2:ix + 1]
                print(f'Time Step {ix + 1}: '
                      f'MAPE {va[0]:7.3%}, {te[0]:7.3%}; '
                      f'MAE  {va[1]:4.3f}, {te[1]:4.3f}; '
                      f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.')
            print(f'Epoch {i:2d} Inference Time {time.time() - start_time:.3f}s')

            if (i + 1) % args.save == 0:
                model_save(sess, global_steps, 'LSGCN', sum_path)
    print('Training model finished!')