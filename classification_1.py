import tensorflow as tf
import numpy as np

def get_dataset() :

	#nombre de ligne par classe
	nblignes = 100

	#Generation des lignes
	sick = np.random.randn(nblignes,2) + np.array([-2,-2])
	sick_2 = np.random.randn(nblignes,2) + np.array([2,2])

	healthy = np.random.randn(nblignes,2) + np.array([2,-2])
	healthy_2 = np.random.randn(nblignes,2) + np.array([-2,2])

	#Mise en forme des donnees
	donnees = np.vstack([sick, sick_2, healthy, healthy_2])
	cibles = np.concatenate((np.zeros(nblignes*2),np.zeros(nblignes*2)+1))
	cibles = cibles.reshape(-1,1)

	return donnees,cibles

if __name__ == '__main__':

	#Recuperation des donnees
	data,target = get_dataset()
	print(data)
	print(target)

	#Affichage des donnees
	#plt.scatter(data[:,0],data[:,1],s=40,c=target)
	#plt.show()

	"""
	 Elements du graphes
	"""

	# Entree
	tf_data = tf.placeholder(tf.float32,shape=[None,2])
	tf_cible = tf.placeholder(tf.float32,shape=[None,1])

	#Variables graphes (poids et biais)
	"""
		1er layer
	"""
	n = 3
	w1 = tf.Variable(tf.random_normal([2,n]))
	b1 = tf.Variable(tf.zeros([n]))

	#Propagation des donnees
	z1 = tf.matmul(tf_data,w1)+b1

	#fonction d'activation du 1er Layer
	py1 = tf.nn.sigmoid(z1)

	"""
		Neurone de sortie
	"""
	w = tf.Variable(tf.random_normal([n,1]))
	b = tf.Variable(tf.zeros([1]))

	#Propagation des donnees
	z = tf.matmul(py1,w)+b

	#fonction d'activation du neuronne de sortie
	py = tf.nn.sigmoid(z)

	"""
		Fonction du graphe
	"""

	#Fonction de cout
	cost = tf.reduce_mean(tf.square(py-tf_cible))

	#Etude de precision
	pred_juste = tf.equal(tf.round(py),tf_cible)
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
			tf_data : data,
			tf_cible: target
			})

		#Calcul precision
		acc = sess.run(accuracy,feed_dict={
			tf_data : data,
			tf_cible: target
			})
		#print("epoch[",e,"]  ","accuracy = ",acc)
