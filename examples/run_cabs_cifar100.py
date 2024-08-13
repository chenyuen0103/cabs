# -*- coding: utf-8 -*-
"""
Run CABS on a CIFAR-10 example.

This will download the dataset to data/cifar-10 automatically if necessary.
"""

import os
import sys
import csv
import time
sys.path.insert(0, os.path.abspath('..'))

import tensorflow as tf
import cifar100_adaptive_batchsize as cifar100
import numpy as np
from cabs import CABSOptimizer
import pdb
import argparse
import time
import random
#### Specify training specifics here ##########################################
from models import cifar100_2conv_3dense as model

num_steps = 8000
learning_rate = 0.1
initial_batch_size = 16
bs_min = 16
bs_max = 2048
###############################################################################


parser = argparse.ArgumentParser(description='CIFAR-10 CABS')
parser.add_argument('--result_dir', type=str, default='./results')
parser.add_argument('--manual_seed', type=int, default=0)
args = parser.parse_args()


random.seed(args.manual_seed)
# Set seed for NumPy
np.random.seed(args.manual_seed)
# Set seed for TensorFlow
tf.set_random_seed(args.manual_seed)


# Set up model
tf.reset_default_graph()
global_bs = tf.Variable(tf.constant(initial_batch_size, dtype=tf.int32))
images, labels = cifar100.inputs(eval_data=False, batch_size=global_bs)
test_images, test_labels = cifar100.inputs(eval_data=True, batch_size=10000)
losses, variables, acc = model.set_up_model(images, labels)
#_, _, test_accuracy = model.set_up_model(test_images, test_labels)

# Set up CABS optimizer
opt = CABSOptimizer(learning_rate, bs_min, bs_max)
sgd_step, bs_new, loss, accuracy = opt.minimize(losses, acc, variables, global_bs)

# Initialize variables and start queues
sess = tf.Session()
coord = tf.train.Coordinator()
sess.run(tf.global_variables_initializer())
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# Open CSV file for logging
csv_file = open(f'{args.result_dir}/cabs_cifar100_s{args.manual_seed}.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Step', 'Loss', 'Batch Size', 'Train Accuracy', 'Train Loss', 'Test Accuracy','Time'])

start_time = time.time()

def evaluate(sess, accuracy_op, test_images_op, test_labels_op):
    """Evaluate the model on test data."""
    test_imgs, test_lbls = sess.run([test_images_op, test_labels_op])
    test_acc = sess.run(accuracy_op, feed_dict={images: test_imgs, labels: test_lbls})
    return test_acc

# Run CABS
for i in range(num_steps):
    _, m_new, l, a = sess.run([sgd_step, bs_new, loss, accuracy])
    print(f'Step {i}: Loss={l}, Batch Size={m_new}, Accuracy={a}')

    if i % 100 == 0:
        # Evaluate test accuracy every 100 steps
        test_acc = evaluate(sess, accuracy, test_images, test_labels)
        print(
            f'Step {i:<4}: Batch Size = {m_new:<5} Train Loss = {l:<12.6f} Test Accuracy = {test_acc:<8.6f}')
        csv_writer.writerow([i, l, m_new, a, test_acc, time.time()-start_time])
    else:
        csv_writer.writerow([i, l, m_new, a, None, time.time()-start_time])

# Compute final test accuracy
final_test_accuracy = evaluate(sess, accuracy, test_images, test_labels)
print(f'Final Test Accuracy: {final_test_accuracy}')
# End timer
end_time = time.time()
total_time = end_time - start_time
print(f'Total Training + Testing Time: {total_time} seconds')
csv_writer.writerow([i, l, m_new, a, final_test_accuracy, total_time])

# Close CSV file
csv_file.close()

# Stop queues
coord.request_stop()
coord.join(threads)