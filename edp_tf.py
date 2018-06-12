import tensorflow as tf
import numpy as np
import gen_data

if __name__ == '__main__':
    #Variable global
    n  = 10000 #Nombres de paquets de données générées
    m  = 500 #Nombres de neurones dans le hidden layer
	#Recuperation des donnees
    train_data,train_target, test_data, test_target = gen_data.generate_data(n)


    """
    Elements du graphes
    """

	# Entree
    tf_data = tf.placeholder(tf.float32,shape=[None,1])
    tf_cible = tf.placeholder(tf.float32,shape=[None,1])

    #Variables graphes (poids et biais)
    """
    	1er layer
    """
    w1 = tf.Variable(tf.random_normal([1,m]))
    b1 = tf.Variable(tf.zeros([m]))

	#Propagation des donnees
    z1 = tf.matmul(tf_data,w1)+b1

    #fonction d'activation du 1er Layer
    py1 = tf.nn.sigmoid(z1)
    """
    	2ieme layer
    """
    w2 = tf.Variable(tf.random_normal([m,m]))
    b2 = tf.Variable(tf.zeros([m]))

	#Propagation despy donnees
    z2 = tf.matmul(z1,w2)+b2

    #fonction d'activation du 1er Layer
    py2 = tf.nn.sigmoid(z2)
    """
    	3ieme layer
    """
    w3 = tf.Variable(tf.random_normal([m,m]))
    b3 = tf.Variable(tf.zeros([m]))

	#Propagation des donnees
    z3 = tf.matmul(z2,w3)+b3

    #fonction d'activation du 1er Layer
    py3 = tf.nn.sigmoid(z3)
    """
    	4ieme layer
    """
    w4 = tf.Variable(tf.random_normal([m,m]))
    b4 = tf.Variable(tf.zeros([m]))

	#Propagation des donnees
    z4 = tf.matmul(z3,w4)+b4

    #fonction d'activation du 1er Layer
    py4 = tf.nn.sigmoid(z4)
    
    """
    	5ieme layer
    """
    w5 = tf.Variable(tf.random_normal([m,m]))
    b5 = tf.Variable(tf.zeros([m]))

	#Propagation des donnees
    z5 = tf.matmul(z4,w5)+b5

    #fonction d'activation du 1er Layer
    py5 = tf.nn.sigmoid(z5)
    """
    Neurone de sortie
    """
    w = tf.Variable(tf.random_normal([m,1]))
    b = tf.Variable(tf.zeros([1]))

    #Propagation des donnees
    z = tf.matmul(py5,w)+b

    #fonction d'activation du neuronne de sortie
    py = tf.nn.sigmoid(z)
    """
    Fonction du graphe
    """
    #Fonction de cout
    cost = tf.reduce_mean(tf.square(py-tf_cible))

    #Etude de precision
    #pred_juste = tf.equal(py,tf_cible)
    pred_juste = tf.less_equal(tf.abs(py-tf_cible),1e-4)
    accuracy = tf.reduce_mean(tf.cast(pred_juste,tf.float32))

    #Entrainement
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.95)
    train = opt.minimize(cost)

    #Creation de la session
    sess = tf.Session()
    #Initialisation des variables
    sess.run(tf.global_variables_initializer())

    ##Execution de l'entrainement
    epoch = 10000
    for e in range(1,epoch+1):
        #Entrainement
        sess.run(train,feed_dict={
        tf_data : train_data,
        tf_cible: train_target
        })
        """
        predicted = sess.run(py, feed_dict={
        tf_data : train_data,
        tf_cible: train_target
        })
        """
        print(sess.run(cost,feed_dict={
        tf_data : train_data,
        tf_cible: train_target
        }))
        #Calcul precision
        """
        acc = sess.run(accuracy,feed_dict={
        tf_data : test_data,
        tf_cible: test_target
        })
        print("epoch[",e,"]  ","accuracy = ",acc)
        """
