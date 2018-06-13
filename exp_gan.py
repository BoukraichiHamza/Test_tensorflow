import tensorflow as tf


"""
Classe de distribution de donnees normales
"""

class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples

"""
Classe de generation de données distribuees normalement et perturbees de
facon aleatoire
"""
        
class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N)+np.random.random(N) * 0.01

"""
Mets en place un layer lineaire et retourne sa sortie
"""

def linear(input, output_dim, scope=None, stddev=1.0):
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'w',
            [input.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
return tf.matmul(input, w) + b

"""
Definition du reseau generator
"""
def generator(input, hidden_size):
    h0 = tf.nn.softplus(linear(input, hidden_size, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1

"""
Definition du reseau discriminator
"""
def discriminator(input, hidden_size):
    h0 = tf.tanh(linear(input, hidden_size * 2, 'd0'))
    h1 = tf.tanh(linear(h0, hidden_size * 2, 'd1'))
    h2 = tf.tanh(linear(h1, hidden_size * 2, 'd2'))
    h3 = tf.sigmoid(linear(h2, 1, 'd3'))
    return h3
