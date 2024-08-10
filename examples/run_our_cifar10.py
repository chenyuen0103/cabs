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
import cifar10_adaptive_batchsize as cifar10

from ours import OurOptimizer

#### Specify training specifics here ##########################################
from models import cifar10_2conv_3dense as model
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




# Total number of training samples
total_samples = 50000

# Generate indices and shuffle
indices = np.arange(total_samples)
np.random.shuffle(indices)

# Split indices into training and validation sets
train_size = int(0.8 * total_samples)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Use the indices in the inputs function
images, labels = cifar10.inputs(eval_data=False, batch_size=global_bs, indices=train_indices)
val_images, val_labels = cifar10.inputs(eval_data=False, batch_size=global_bs, indices=val_indices)



test_images, test_labels = cifar10.inputs(eval_data=True, batch_size=10000)
losses, variables, acc = model.set_up_model(images, labels)
#_, _, test_accuracy = model.set_up_model(test_images, test_labels)

# Set up CABS optimizer
opt = OurOptimizer(learning_rate, bs_min, bs_max)
sgd_step, bs_new, grad_div, loss, accuracy = opt.minimize(losses, acc, variables, global_bs)

# Initialize variables and start queues
sess = tf.Session()
coord = tf.train.Coordinator()
sess.run(tf.global_variables_initializer())
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# Open CSV file for logging
csv_file = open('our_cifar10_training_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Step', 'Gradient Diversity', 'Batch Size', 'Train Loss','Train Accuracy', 'Val Accuracy', 'Test Accuracy', 'Time'])

start_time = time.time()

def evaluate(sess, accuracy_op, test_images_op, test_labels_op):
    """Evaluate the model on test data."""
    test_imgs, test_lbls = sess.run([test_images_op, test_labels_op])
    test_acc = sess.run(accuracy_op, feed_dict={images: test_imgs, labels: test_lbls})
    return test_acc



m_new = initial_batch_size
# Run CABS
for i in range(num_steps):
    # _, m_new, l, a = sess.run([sgd_step, bs_new, loss, accuracy])
    m_used = m_new
    _, m_new, gd, l, a = sess.run([sgd_step, bs_new, grad_div, loss, accuracy])
    # _, m_new, gd, l, a = sess.run([sgd_step, bs_new, grad_div, loss, accuracy],
    #                               feed_dict={images: images, labels: labels})
    # print(f'Step {i}: Loss={l}, Batch Size={m_new}, Accuracy={a}')

    if i % 100 == 0:
        # Evaluate test accuracy every 100 steps
        # val_acc = evaluate(sess, accuracy, val_images, val_labels)
        # val_imgs, val_lbls = sess.run([val_images, val_labels])
        # Now use the actual data in the feed_dict
        # val_acc = sess.run(accuracy)
        # test_acc = sess.run(accuracy, feed_dict={images: test_images, labels: test_labels})
        test_acc = evaluate(sess, accuracy, test_images, test_labels)
        print(f'Step {i:<4}: Grad_Div = {gd:<10.4f}, Batch Size = {m_used:<5} Train Loss = {l:<12.6f} Test Accuracy = {test_acc:<8.6f}')
        csv_writer.writerow([i, grad_div, m_used, l,  a, None, test_acc, time.time() - start_time])
    else:
        csv_writer.writerow([i, grad_div, m_used, l,  a, None, None, time.time() - start_time])

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