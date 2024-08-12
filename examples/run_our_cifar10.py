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
import numpy as np
import tensorflow as tf
# import cifar10_adaptive_batchsize_split as cifar10
import cifar10_adaptive_batchsize as cifar10

from ours import OurOptimizer
import argparse
#### Specify training specifics here ##########################################
from models import cifar10_2conv_3dense as model
import pdb
import random
# Set seed for Python's built-in random module



parser = argparse.ArgumentParser(description='CIFAR-10 CABS')
parser.add_argument('--delta', type=float, default=1)
parser.add_argument('--result_dir', type=str, default='./results')
parser.add_argument('--manual_seed', type=int, default=0)
args = parser.parse_args()

delta = args.delta
num_steps = 8000
learning_rate = 0.1
initial_batch_size = 16
bs_min = 16
bs_max = 2048
validation_split = 0.2  # 20% for validation
buffer_size = 50000  # Size of the dataset for shuffling
###############################################################################

# # Set up model
tf.reset_default_graph()
global_bs = tf.Variable(tf.constant(initial_batch_size, dtype=tf.int32))
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)




# Total number of training samples
total_samples = 50000

random.seed(args.manual_seed)
# Set seed for NumPy
np.random.seed(args.manual_seed)
# Set seed for TensorFlow
tf.set_random_seed(args.manual_seed)


# images, labels, num_examples, total_examples  = cifar10.inputs(eval_data=False, batch_size=global_bs, use_holdout=False)
images, labels= cifar10.inputs(eval_data=False, batch_size=global_bs)
# Load the validation dataset
# val_images, val_labels, val_num_examples, val_total_examples = cifar10.inputs(eval_data=False, batch_size=10000, use_holdout=True)
# test_images, test_labels, test_num_examples, test_total_examples = cifar10.inputs(eval_data=True, batch_size=10000)
test_images, test_labels = cifar10.inputs(eval_data=True, batch_size=10000)

# Set up the model for training data
losses, variables, acc = model.set_up_model(images, labels)

# Set up the model for validation data (optional, for tracking validation accuracy)
# val_losses, _, val_acc = model.set_up_model(val_images, val_labels)
#
# test_losses, _, test_accuracy = model.set_up_model(test_images, test_labels)

# Set up CABS optimizer
opt = OurOptimizer(learning_rate, bs_min, bs_max)
sgd_step, bs_new, grad_div, loss, accuracy = opt.minimize(losses, acc, variables, global_bs, delta=delta)


# Initialize variables and start queues
sess = tf.Session()
coord = tf.train.Coordinator()
sess.run(tf.global_variables_initializer())
threads = tf.train.start_queue_runners(sess=sess, coord=coord)



# Open CSV file for logging
csv_file = open(f'{args.result_dir}/our_cifar10_delta{delta}_s{args.manual_seed}.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
# csv_writer.writerow(['Step', 'Gradient Diversity', 'Batch Size', 'Train Loss','Train Accuracy', 'Val Accuracy', 'Test Accuracy', 'Time'])
csv_writer.writerow(['Step', 'Gradient Diversity', 'Batch Size', 'Train Loss','Train Accuracy', 'Test Accuracy', 'Time'])


start_time = time.time()

def evaluate(sess, accuracy_op, test_images_op, test_labels_op):
    """Evaluate the model on test data."""
    test_imgs, test_lbls = sess.run([test_images_op, test_labels_op])
    test_acc = sess.run(accuracy_op, feed_dict={images: test_imgs, labels: test_lbls})
    return test_acc


# actual_num_examples = sess.run(num_examples)
# print(f"Number of examples in this batch: {actual_num_examples}")
# print(f"Total number of examples in the dataset: {total_examples}")
#
# val_num_examples = sess.run(val_num_examples)
#
# print(f"Number of examples in the validation set: {val_num_examples}"
#       f"\nTotal number of examples in the validation set: {val_total_examples}")
#
# test_num_examples = sess.run(test_num_examples)
# print(f"Number of examples in the test set: {test_num_examples}"
#       f"\nTotal number of examples in the test set: {test_total_examples}")

m_new = initial_batch_size
# Run CABS
for i in range(num_steps):

    # _, m_new, l, a = sess.run([sgd_step, bs_new, loss, accuracy])
    m_used = m_new
    _, m_new, gd, l, a = sess.run([sgd_step, bs_new, grad_div, loss, accuracy])

    if i % 100 == 0:
        # Evaluate test accuracy every 100 steps
        # val_acc = evaluate(sess, accuracy, val_images, val_labels)
        test_acc = evaluate(sess, accuracy, test_images, test_labels)
        print(f'Step {i:<4}: Grad_Div = {gd:<10.4f}, Batch Size = {m_used:<5} Train Loss = {l:<12.6f} Train Acc = {a:<12.6f} Test Accuracy = {test_acc:<8.6f}')
        csv_writer.writerow([i, gd, m_used, l,  a, test_acc, time.time() - start_time])
    else:
        csv_writer.writerow([i, gd, m_used, l,  a, None, time.time() - start_time])

    csv_file.flush()


# val_acc = evaluate(sess, accuracy, val_images, val_labels)
test_acc = evaluate(sess, accuracy, test_images, test_labels)
print(f'Step {i:<4}: Grad_Div = {gd:<10.4f}, Batch Size = {m_used:<5} Train Loss = {l:<12.6f} Train Acc = {a:<12.6f} Test Accuracy = {test_acc:<8.6f}')
csv_writer.writerow([i, gd, m_used, l,  a, None, time.time() - start_time])
# Compute final test accuracy
# Compute final test accuracy
final_test_accuracy = evaluate(sess, accuracy, test_images, test_labels)
print(f'Final Test Accuracy: {final_test_accuracy}')
# End timer
end_time = time.time()
total_time = end_time - start_time
print(f'Total Training + Testing Time: {total_time} seconds')

# Close CSV file
csv_file.close()

# Stop queues
coord.request_stop()
coord.join(threads)


