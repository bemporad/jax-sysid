"""
Use jax-sysid and Keras to solve Fashion MNIST image classification problem. 

To run this file, you must have Keras installed with KERAS_BACKEND="jax".

Cf. https://www.tensorflow.org/tutorials/keras/classification

(C) 2024 A. Bemporad, May 7, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
from jax_sysid.models import StaticModel
from jax_sysid.utils import compute_scores
from jax import nn
import jax.numpy as jnp

np.random.seed(1)
plt.ion()

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# Scale images to the [0, 1] range and one-hot encode labels
Us_train=(train_images/255.0).reshape(-1,28*28)
Ys_train = keras.utils.to_categorical(train_labels) # one-hot encoding
Us_test=(test_images/255.0).reshape(-1,28*28)
Ys_test = keras.utils.to_categorical(test_labels)

# Define model
keras_model = keras.Sequential([
    keras.layers.Input((28,28)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='tanh'),
    keras.layers.Dense(64, activation='tanh'),
    keras.layers.Dense(32, activation='tanh'),
    #keras.layers.Dense(10, activation='softmax') # better introduce softmax in the loss function
    keras.layers.Dense(10)
])

def output_fcn(u,params):
    keras_model.set_weights(params)
    return keras_model(u.reshape(-1,28,28))

model = StaticModel(10,28*28,output_fcn=output_fcn)

# Train model
params=keras_model.get_weights()
model.init(params=params)

#cross_entropy = None # Default: MSE loss
# def cross_entropy(Yhat,Y):
#     return jnp.sum(-Y*jnp.log(Yhat)) # cross-entropy loss when softmax is used in the output layer
def cross_entropy(Yhat,Y):
    return -jnp.sum(nn.log_softmax(Yhat, axis=1)*Y)/Y.shape[0] # cross-entropy loss
model.loss(rho_th=1.e-4, output_loss=cross_entropy)
model.optimization(adam_epochs=0, lbfgs_epochs=2000, iprint=10) 
model.fit(Ys_train, Us_train)
t0 = model.t_solve

print(f"Elapsed time: {t0} s")
Yhat_train = np.argmax(model.predict(Us_train),axis=1)
Yhat_test = np.argmax(model.predict(Us_test),axis=1)

acc, acc_test, msg = compute_scores(
    train_labels, Yhat_train, test_labels, Yhat_test, fit='accuracy')

print(msg)
