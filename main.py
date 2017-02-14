import tensorflow as tf
import numpy as np

import edward as ed
from edward.models import Bernoulli, Normal, Beta, Dirichlet


sess = ed.get_session()

A_data = Normal(mu=0.5, sigma=1.)
B_data = Normal(mu=-0.2, sigma=1.)
C_data = Normal(mu=tf.add(A_data, B_data), sigma=0.05)

def sample_joint():
    return sess.run([A_data.value(), B_data.value(), C_data.value()])

dataset = tf.constant([sample_joint() for i in range(20)])

Dirichlet(tf.ones(5)*20).sample().eval()

connections_binary = Bernoulli(tf.Variable([0.5, 0.5, 0.5]))
connections_weight = Normal(mu=tf.Variable(tf.zeros(3)),
                            sigma=tf.Variable(tf.ones(3)))

# 1 -> 2, 1 -> 3, 2 -> 3
directions_binary = Bernoulli(tf.Variable([0.5, 0.5, 0.5]))

tf.cond(tf.constant(5) < tf.constant(3),
        lambda: tf.ones(1),
        lambda: tf.zeros(1)).eval()
