import time

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
import matplotlib.pyplot as plt

from utils import get_minibatches_idx

# Based on https://jmetzen.github.io/2015-11-27/vae.html
# and https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder_deconv.py

n_samples = 60000
eps = 1e-10
img_size = 28
num_channels = 1
num_filters = 16
embedding_size = 128
batch_size = 100
num_epochs = 1

# To do:
# Evaluate on the validation set after each epoch


class VariationalAutoencoder(object):
	# See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details
	
	def __init__(self):
		self.x = tf.placeholder(tf.float32, [None, img_size, img_size, num_channels])

		# Encoder
		x = Reshape([img_size*img_size])(self.x)
		h = Dense(500, activation='relu')(x)
		h = Dense(500, activation='relu')(h)
		
		# Encode as mean and variance vectors
		self.z_mean = Dense(embedding_size)(h)
		self.z_log_sigma_sq = Dense(embedding_size)(h)

		# Draw one sample z from Gaussian distribution
		noise = tf.random_normal((batch_size, embedding_size), 0, 1, dtype=tf.float32)
		
		# Add noise
		self.z = self.z_mean + noise*tf.sqrt(tf.exp(self.z_log_sigma_sq))

		# Decoder		
		h = Dense(500, activation='relu')(self.z)
		h = Dense(500, activation='relu')(h)
		h = Dense(500, activation='relu')(h)
		h = Dense(img_size*img_size, activation='sigmoid')(h)
		self.x_reconstr_mean = Reshape([img_size,img_size,1])(h)
		print(self.x_reconstr_mean)
		reconstr_loss = -tf.reduce_sum(self.x * tf.log(eps + self.x_reconstr_mean)
						   + (1-self.x) * tf.log(eps + 1 - self.x_reconstr_mean), [1,2,3])
			
		# KL-divergence
		latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
										   - tf.square(self.z_mean) 
										   - tf.exp(self.z_log_sigma_sq), 1)
										   
		self.cost = tf.reduce_mean(reconstr_loss + latent_loss) # average over batch

		self.train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())


	def generate(self, z_mu=None):
		if z_mu is None:
			z_mu = [np.random.normal(size=embedding_size) for i in range(batch_size)]			
		return self.sess.run(self.x_reconstr_mean, feed_dict={self.z: z_mu})
	

	def train(self, X):
		print("\nStarting training")
		
		for epoch in range(num_epochs):
			avg_cost = 0.0
			train_indices = get_minibatches_idx(len(X), batch_size, shuffle=True)

			for it in train_indices:
				batch_x = [X[i] for i in it]
				_, cost = self.sess.run((self.train_step, self.cost), feed_dict={self.x: batch_x})
				
				avg_cost += cost / n_samples * batch_size

			print("Epoch:", '%d' % (epoch+1), "cost=", "{:.3f}".format(avg_cost))


def main():
	# Load the data
	# It will be downloaded first if necessary
	(X_train, _), (X_test, _) = mnist.load_data()
	
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train = np.reshape(X_train,[-1,img_size,img_size,num_channels])
	X_test = np.reshape(X_test,[-1,img_size,img_size,num_channels])
	X_train /= 255
	X_test /= 255
	
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	vae = VariationalAutoencoder()
	print("Model compiled")			 
			 
	vae.train(X_train)

	x_sample = X_test[:100]	
	x_gen = vae.generate()
	
	plt.figure(figsize=(8,12))
	
	# Show 5 generated images and 5 images from the test set
	for i in range(5):
		plt.subplot(5, 2, 2*i + 1)
		plt.imshow(np.squeeze(x_sample[i]), vmin=0, vmax=1, cmap="gray")
		plt.title("Test image")
		
		plt.subplot(5, 2, 2*i + 2)
		plt.imshow(np.squeeze(x_gen[i]), vmin=0, vmax=1, cmap="gray")
		plt.title("Generated image")
		
	plt.tight_layout()
	plt.show()
	
if __name__ == "__main__":
	main()
