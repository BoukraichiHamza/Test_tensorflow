import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Recuperation des donnees
mnist = input_data.read_data_sets("input/data", one_hot=True)

# Entree du graphe
tf_data = tf.placeholder(tf.float32, [None, 784])
tf_cible = tf.placeholder(tf.float32, [None, 10])

""" 
	Neurones de sorties
"""

# Poids et biais
w1 = tf.Variable(tf.random_normal([784, 10]))
b1 = tf.Variable(tf.zeros([10]))

#  Propagation des donnees
z1 = tf.matmul(tf_data, w1) + b1

# Normalisation des donnees
softmax = tf.nn.softmax(z1)
"""
	Error + Train
"""

# calcul de l'erreur selon l'entropy softmax
error = tf.nn.softmax_cross_entropy_with_logits(labels=tf_cible, logits=z1)

# Entrainement
train = tf.train.GradientDescentOptimizer(0.5).minimize(error)

# precision
pred_juste = tf.equal(tf.argmax(softmax, 1), tf.argmax(tf_cible, 1))
accuracy = tf.reduce_mean(tf.cast(pred_juste, tf.float32))

""" Lancement entrainement et calcul de la precision """

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Train the model
    epochs = 10000
    for e in range(epochs):
        batch_data, batch_cible = mnist.train.next_batch(100)
        sess.run(train, feed_dict={tf_data: batch_data, tf_cible: batch_cible})

    true_cls = []
    py_cls = []
    for c in range(0, 20):
        py = sess.run(softmax, feed_dict={
            tf_data: [mnist.test.images[c]]
        })
        true_cls.append(np.argmax(mnist.test.labels[c]))
        py_cls.append(np.argmax(py))
    print("true_cls", true_cls)
    print("py cls", py_cls)

    # Accuracy on the test set
    acc = sess.run(accuracy, feed_dict={
        tf_data: mnist.test.images,
        tf_cible:  mnist.test.labels
    })
    print("Accuracy on the test set", acc)


