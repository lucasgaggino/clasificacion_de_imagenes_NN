import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from keras.utils import plot_model
from tensorflow.python.util import object_identity

img_dir = './../../../Neuronales/mono/'

def count_params(weights):
    return int(sum(np.prod(p.shape.as_list())
                   for p in object_identity.ObjectIdentitySet(weights)))


def trainable_params(model):
    if hasattr(model, '_collected_trainable_weights'):
        return count_params(model._collected_trainable_weights)
    else:
        return model.trainable_weights


def red_cnn_t1(params):
    if params[3] != 0:
        layer = [
            tf.keras.layers.Conv2D(filters=params[0], kernel_size=(3, 3), padding="same", activation=tf.nn.relu,
                                   input_shape=train_images.shape[1:]),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Conv2D(filters=params[1], kernel_size=(3, 3), padding="same", activation=tf.nn.relu,
                                   input_shape=train_images.shape[1:]),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=params[2], activation=tf.nn.relu),
            tf.keras.layers.Dense(units=params[3], activation=tf.nn.relu),
            tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
        ]
    else:
        layer = [
            tf.keras.layers.Conv2D(filters=params[0], kernel_size=(3, 3), padding="same", activation=tf.nn.relu,
                                   input_shape=train_images.shape[1:]),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Conv2D(filters=params[1], kernel_size=(3, 3), padding="same", activation=tf.nn.relu,
                                   input_shape=train_images.shape[1:]),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=params[2], activation=tf.nn.relu),
            tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
        ]
    return layer


def red_cnn_t2(params):
    if params[4] != 0:
        layer = [
            tf.keras.layers.Conv2D(filters=params[0], kernel_size=(3, 3), padding="same", activation=tf.nn.relu,
                                   input_shape=train_images.shape[1:]),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Conv2D(filters=params[1], kernel_size=(3, 3), padding="same", activation=tf.nn.relu,
                                   input_shape=train_images.shape[1:]),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Conv2D(filters=params[2], kernel_size=(3, 3), padding="same", activation=tf.nn.relu,
                                   input_shape=train_images.shape[1:]),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=params[3], activation=tf.nn.relu),
            tf.keras.layers.Dense(units=params[4], activation=tf.nn.relu),
            tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
        ]
    else:
        layer = [
            tf.keras.layers.Conv2D(filters=params[0], kernel_size=(3, 3), padding="same", activation=tf.nn.relu,
                                   input_shape=train_images.shape[1:]),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Conv2D(filters=params[1], kernel_size=(3, 3), padding="same", activation=tf.nn.relu,
                                   input_shape=train_images.shape[1:]),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Conv2D(filters=params[2], kernel_size=(3, 3), padding="same", activation=tf.nn.relu,
                                   input_shape=train_images.shape[1:]),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=params[3], activation=tf.nn.relu),
            tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
        ]
    return layer


def red_dense(params):
    if params[1] != 0:
        layer = [
            tf.keras.layers.Flatten(input_shape=train_images.shape[1:]),
            tf.keras.layers.Dense(params[0], activation='relu'),
            tf.keras.layers.Dense(params[1], activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ]
    else:
        layer = [
            tf.keras.layers.Flatten(input_shape=train_images.shape[1:]),
            tf.keras.layers.Dense(params[0], activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ]
    return layer


def acc_complexity(layers):
    complexity = []
    accuracy = []
    for exp in layers:
        red1 = tf.keras.Sequential(exp)
        red1.compile(optimizer=tf.optimizers.Adam(),
                     loss=tf.losses.SparseCategoricalCrossentropy(),
                     metrics=[tf.metrics.SparseCategoricalAccuracy()])
    # entrenamiento
        red1.fit(train_images, train_labels, epochs=5, batch_size=50)
        test_loss, test_acc = red1.evaluate(test_images, test_labels, verbose=2)
        complexity.append(trainable_params(red1))
        accuracy.append(test_acc)
    plot_accuracy = [x for _,x in sorted(zip(complexity, accuracy))]
    complexity = sorted(complexity)
    return complexity, plot_accuracy


# cargo el dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# pre-procesamiento de las imgs
train_images = train_images / 255.0
test_images = test_images / 255.0
# expando las dims de las imgs para que sean tensores
train_images = np.expand_dims(train_images, axis=-1)
train_labels = np.array(train_labels)
test_images = np.expand_dims(test_images, axis=-1)
test_labels = np.array(test_labels)

# diseno las capas
layersCNN1 = [
    red_cnn_t1([1, 2, 12, 0]),
    red_cnn_t1([10, 15, 20, 0]),
    red_cnn_t1([2, 4, 12, 0]),
    red_cnn_t1([1, 2, 15, 0]),
    red_cnn_t1([2, 4, 20, 12]),
    red_cnn_t1([1, 2, 30, 0]),
    red_cnn_t1([1, 2, 32, 0]),
    red_cnn_t1([5, 10, 12, 0]),
    red_cnn_t1([5, 10, 12, 10]),
    red_cnn_t1([2, 4, 10, 10]),
    red_cnn_t1([10, 20, 20, 12]),
    red_cnn_t1([16, 32, 15, 12])
]

layersCNN2 = [
    red_cnn_t2([1, 1, 2, 12, 0]),
    red_cnn_t2([1, 2, 4, 12, 0]),
    red_cnn_t2([2, 4, 8, 12, 0]),
    red_cnn_t2([3, 6, 12, 12, 0]),
    red_cnn_t2([1, 1, 2, 15, 0]),
    red_cnn_t2([2, 4, 8, 12, 10]),
    red_cnn_t2([3, 6, 12, 12, 10]),
    red_cnn_t2([16, 32, 64, 12, 0]),
    red_cnn_t2([16, 32, 64, 8, 0]),
    red_cnn_t2([4, 16, 32, 15, 0])
]

layersD = [
    red_dense([8, 0]),
    red_dense([10, 0]),
    red_dense([12, 0]),
    red_dense([5, 0]),
    red_dense([7, 0]),
    red_dense([12, 0]),
    red_dense([15, 0]),
    red_dense([20, 0]),
    red_dense([30, 0]),
    red_dense([32, 0]),
]

layersD2 = [
    red_dense([8, 8]),
    red_dense([10, 10]),
    red_dense([12, 5]),
    red_dense([5, 5]),
    red_dense([7, 8]),
    red_dense([12, 10]),
    red_dense([15, 10]),
    red_dense([20, 10]),
    red_dense([30, 15]),
    red_dense([32, 16]),
]

# creo los modelos


complexity_plotCNN1, accuracy_plotCNN1 = acc_complexity(layersCNN1)
complexity_plotCNN2, accuracy_plotCNN2 = acc_complexity(layersCNN2)
complexity_plotD1, accuracy_plotD1 = acc_complexity(layersD)
complexity_plotD2, accuracy_plotD2 = acc_complexity(layersD2)

plt.plot(complexity_plotCNN1, accuracy_plotCNN1, marker='h', label='Red CNN 2 filtros', color='steelblue')
plt.plot(complexity_plotCNN2, accuracy_plotCNN2, marker='H', label='Red CNN 3 filtros', color='navy')
plt.plot(complexity_plotD1, accuracy_plotD1, marker='d', label='Red Densa 2 capas', color='red')
plt.plot(complexity_plotD2, accuracy_plotD2, marker='D', label='Red Densa 3 capas', color='maroon')


plt.title('Desempeño de Redes Neuronales')
plt.ylabel('Precisión')
plt.xlabel('Parámetros entrenables')
# plt.axis([0,1000,0.8,1])
plt.grid()
plt.legend(loc="best")
plt.savefig(img_dir + 'exp1.png')
plt.show()
