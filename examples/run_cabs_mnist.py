# -*- coding: utf-8 -*-
"""
Run CABS on a MNIST example.

This will download the dataset to data/mnist automatically if necessary.
"""

import os
import sys
import csv
import time
sys.path.insert(0, os.path.abspath('..'))

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

from cabs import CABSOptimizer

#### Specify training specifics here ##########################################
from models import mnist_2conv_2dense as model
num_steps = 2000
learning_rate = 0.1
initial_batch_size = 16
bs_min = 16
bs_max = 2048
###############################################################################

# Set up model
losses, placeholders, variables, acc = model.set_up_model()
X, y = placeholders

# Set up CABS optimizer
opt = CABSOptimizer(learning_rate, bs_min, bs_max)
sgd_step, bs_new, loss, accuracy = opt.minimize(losses, acc, variables)

# Initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Open CSV file for logging
csv_file = open('cabs_training_mnist_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Step', 'Loss', 'Batch Size', 'Train Accuracy', 'Test Accuracy'])

start_time = time.time()

# Run CABS
m = initial_batch_size
for i in range(num_steps):
  batch = mnist.train.next_batch(m)
  _, m_new, l, a = sess.run([sgd_step, bs_new, loss, accuracy], {X: batch[0], y: batch[1]})
  print(f'Step {i}: Loss={l}, Batch Size={m_new}, Accuracy={a}')

  # Evaluate test accuracy every 100 steps
  if i % 100 == 0:
      test_accuracy = sess.run(accuracy, {X: mnist.test.images, y: mnist.test.labels})
      print(f'Step {i}: Test Accuracy={test_accuracy}')
      csv_writer.writerow([i, l, m_new, a, test_accuracy])
  else:
      csv_writer.writerow([i, l, m_new, a, None])
  m = m_new

# Compute final test accuracy
final_test_accuracy = sess.run(accuracy, {X: mnist.test.images, y: mnist.test.labels})
print(f'Final Test Accuracy: {final_test_accuracy}')

# End timer
end_time = time.time()
total_time = end_time - start_time
print(f'Total Training + Testing Time: {total_time} seconds')