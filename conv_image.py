import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

"""
    Récupération des jeux de données
"""

files = glob.glob("train/*.jpg")

# Structuration des données
targets = []
features = []
for file in files:
    features.append(np.array(Image.open(file).resize((75,75))))
    # Label : 0 pour chat, 1 pour chien
    if  "cat" in file:
        target = [0,1]
    else:
        target = [1,0]
    targets.append(target)
    print(file)

features = np.array(features)
targets = np.array(targets)
# print(features.shape)
# print(targets.shape)

# Separation en jeu d'entrainement et jeu de validation
X_train,X_valid,y_train,y_valid = train_test_split(features,targets,test_size=0.1,random_state=42)
# print(X_train.shape)
# print(X_valid.shape)
# print(y_train.shape)
# print(y_valid.shape)

"""
    Variables du graphe
"""
#placeholder
x = tf.placeholder(tf.float32,(None,75,75,3))
y = tf.placeholder(tf.float32,(None,2))

def create_conv(prev,filter_size,nb_filter):
    #Filtres
    w_filters = tf.Variable(tf.truncated_normal(shape=(filter_size,filter_size,int(prev.get_shape()[-1]),nb_filter)))

    b_filters = tf.Variable(tf.zeros(shape=nb_filter))

    #Convolutions
    conv = tf.nn.conv2d(prev,w_filters,strides=[1,1,1,1],padding="SAME") + b_filters
    #Fontion d'Activation
    conv = tf.nn.relu(conv)
    #Pooling
    conv = tf.nn.max_pool(conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    return conv

#Création des Convolutions
conv = create_conv(x,8,32)
conv = create_conv(conv,5,64)
conv = create_conv(conv,5,128)
conv = create_conv(conv,5,256)
conv = create_conv(conv,5,215)
flatconv = tf.contrib.layers.flatten(conv)

# First layer
w1 = tf.Variable(tf.truncated_normal(shape=(int(flatconv.get_shape()[1]),521)))
b1 = tf.Variable(tf.zeros(521))

fc1 = tf.matmul(flatconv,w1) + b1
fc1 = tf.nn.relu(fc1)

# Output layer
w2 = tf.Variable(tf.truncated_normal(shape=(521,2)))
b2 = tf.Variable(tf.zeros(2))

fc2 = tf.matmul(fc1,w2) + b2
fc2 = tf.nn.relu(fc2)

softmax = tf.nn.softmax(fc2)

"""
    Fonction du graphe
"""
# Error
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits = fc2)
loss_operation = tf.reduce_mean(cross_entropy)
#Accurary
print(softmax.shape)
correct_prediction = tf.equal(tf.argmax(softmax,axis=1),tf.argmax(y,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#Optimizer
optimizer = tf.train.AdamOptimizer(0.0001)
train_op = optimizer.minimize(loss_operation)

"""
    Entrainement du modele
"""
batch_size = 100 # taille batch d'entrainement
sess = tf.Session() # lancement de la session tensorflow
#Initialisation des donnees
sess.run(tf.global_variables_initializer())
epochs = 10000 # Nombre d'epoch d'Entrainement

for epoch in range(0,epochs):
    #Reoordonnancement random des images
    index=np.arange(len(X_train))
    np.random.shuffle(index)
    X_train = X_train[index]
    y_train = y_train[index]

    #Entrainement sur plusieurs batch
    for ind in range(0,len(X_train),batch_size):
        #Entrainement
        batch = X_train[ind:ind+batch_size]
        sess.run(train_op, feed_dict={
        x : batch,
        y : y_train[ind:ind+batch_size]
        })

    #Calcul de la précision
    accs = []
    for ind in range(0,len(X_train),batch_size):
        #Precision sur un batch de test
        batch = X_train[ind:ind+batch_size]
        acc = sess.run(accuracy, feed_dict={
        x : batch,
        y : y_train[ind:ind+batch_size]
        })
        accs.append(acc)

    print("epoch[",epoch,"] accuracy = ",np.mean(accs))
