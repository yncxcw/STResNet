'''
Author: Sneha Singhania

This file contains the main program. The computation graph for ST-ResNet is built, launched in a session and trained here.
'''

from st_resnet import Graph
import tensorflow as tf
from params import Params as param
from tqdm import tqdm
from utils import batch_generator
import numpy as np
import h5py
from dataloader import  STRdataloader

if __name__ == '__main__': 
    # build dataloader
    dataloader = STRdataloader()
    # build the computation graph
    g = Graph()
    print ("Computation graph for ST-ResNet loaded\n")
    # create summary writers for logging train and test statistics
    train_writer = tf.summary.FileWriter('./logdir/train', g.loss.graph)
    val_writer = tf.summary.FileWriter('./logdir/val', g.loss.graph)   
        
    train_batch_generator = batch_generator(dataloader, param.batch_size, "train")
    test_batch_generator = batch_generator(dataloader, param.batch_size, "test")

    num_train_batch = int(0.7 * len(dataloader)) // param.batch_size
    num_test_batch = int(0.3 * len(dataloader)) // param.batch_size

    print("Start learning:")
    epoch_output = []
    epoch_gt = []
    with tf.Session(graph=g.graph) as sess:
        sess.run(tf.global_variables_initializer())        
        for epoch in range(param.num_epochs):            
            loss_train = 0
            loss_val = 0
            print("Epoch: {}\t".format(epoch), )
            # training
            for b in tqdm(range(num_train_batch)):
                x_closeness, x_period, x_trend, y_batch = next(train_batch_generator)
                loss_tr, _, summary = sess.run([g.loss, g.optimizer, g.merged],
                                                feed_dict={g.c_inp: x_closeness,
                                                           g.p_inp: x_period,
                                                           g.t_inp: x_trend,
                                                           g.output: y_batch})
                loss_train = loss_tr * param.delta + loss_train * (1 - param.delta)
                train_writer.add_summary(summary, b + num_train_batch * epoch)

            # testing
            batch_output = []
            batch_gt = []
            for b in tqdm(range(num_test_batch)):
                x_closeness, x_period, x_trend, y_batch = next(test_batch_generator)                
                loss_v, output, summary = sess.run([g.loss, g.output, g.merged],
                                            feed_dict={g.c_inp: x_closeness,
                                                       g.p_inp: x_period,
                                                       g.t_inp: x_trend,
                                                       g.output: y_batch})
                loss_val += loss_v
                val_writer.add_summary(summary, b + num_test_batch * epoch)
                
                # prediction output [batch_size, H, W, 1] 
                denormalized_output = dataloader.inverse_transform(output)
                batch_output.append(denormalized_output)
                batch_gt.append(y_batch)

            batch_output = np.stack(batch_output, axis=0)
            batch_gt = np.stack(batch_gt, axis=0)
            if num_test_batch != 0:
                loss_val /= num_test_batch

            print("loss: {:.3f}, val_loss: {:.3f}".format(loss_train, loss_val))  
            # save the model after every epoch         
            g.saver.save(sess, "/tmp/model")
            epoch_output.append(batch_output)
            epoch_gt.append(batch_gt)

    epoch_output = np.stack(epoch_output, axis=0)
    epoch_gt = np.stack(epoch_gt, axis=0)
    np.save("outputs", epoch_output)
    np.save("gts", epoch_gt)
    train_writer.close()
    val_writer.close()
    print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")
