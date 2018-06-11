import tensorflow as tf
import numpy as np
import gen_data

if __name__ == '__main__':
    #Variable global
    n  = 1000 #Nombres de paquets de données générées
    m  = 10  #Nombres de données générées par paquet
    maux = 5 # m/2
    r1 = 20  #Nombres de noeuds en sortie du premier layer
    r2 = 10  #Nombres de noeuds en sortie du second layer
	#Recuperation des donnees
    train_data,train_target, test_data, test_target = gen_data.generate_data(maux,n)


    """
    Elements du graphes
    """

	# Entree
    tf_data = tf.placeholder(tf.float32,shape=[None,m])
    tf_cible = tf.placeholder(tf.float32,shape=[None,1])

    #Variables graphes (poids et biais)
    """
    	1er layer
    """
    w1 = tf.Variable(tf.random_normal([m,r1]))
    b1 = tf.Variable(tf.zeros([r1]))

	#Propagation des donnees
    z1 = tf.matmul(tf_data,w1)+b1

    #fonction d'activation du 1er Layer
    py1 = tf.nn.sigmoid(z1)

    """
    	2ieme layer
    """
    w2 = tf.Variable(tf.random_normal([r1,r2]))
    b2 = tf.Variable(tf.zeros([r2]))

	#Propagation des donnees
    z2 = tf.matmul(py1,w2)+b2

    #fonction d'activation du 1er Layer
    py2 = tf.nn.sigmoid(z2)

    """
    Neurone de sortie
    """
    w = tf.Variable(tf.random_normal([r2,1]))
    b = tf.Variable(tf.zeros([1]))

    #Propagation des donnees
    z = tf.matmul(py2,w)+b

    #fonction d'activation du neuronne de sortie
    py = tf.nn.sigmoid(z)
    """
    Fonction du graphe
    """
    #Fonction de cout
    cost = tf.reduce_mean(tf.square(py-tf_cible))

    #Etude de precision
    #pred_juste = tf.equal(py,tf_cible)
    pred_juste = tf.less_equal(tf.abs(py-tf_cible),1e-1)
    accuracy = tf.reduce_mean(tf.cast(pred_juste,tf.float32))

    #Entrainement
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
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
        #Calcul precision
        acc = sess.run(accuracy,feed_dict={
        tf_data : train_data,
        tf_cible: train_target
        })
        print("epoch[",e,"]  ","accuracy = ",acc)
