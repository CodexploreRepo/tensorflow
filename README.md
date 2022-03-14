# Deep Learning
# Advanced Machine Learning

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Tensorflow Modules](#1-tensorflow-modules)
  - [1.1. Keras](#11-keras)  
- [2. Model Creation](#2-model-creation)
  - [2.1. Sequential API](#21-sequential-api)
  - [2.2. Functional API](#22-functional-api)
  - [2.3. Model Training](#23-model-training)   


[(Back to top)](#table-of-contents)

# 1. Tensorflow Modules
```Python
#Tensorflow
import tensorflow as tf

tf.feature_column. #Feature Columns
```
## 1.1. Keras 
```Python
#Keras
from tensorflow import keras

#Datasets
keras.datasets.mnist.load_data()

#Model Creation 
keras.Input(shape=(32, 32, 3)) #Input shape
keras.layers. #Keras Layers


#Utils
#To plot the model structure 
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
```

# 2. Model Creation
## 2.1. Sequential API
- There are 2 ways to create a model using Sequential API
#### 2.1.1. Method 1: using `add` method
```Python
'''
Create a NN with 1 input layer
2 Dense hidden layer with 64 nodes each, activation function reLU for each node
1 output layer with 10 outputs (interpreted as probabilites over 10 classes)
'''
'''Read more about Sequential API here: https://keras.io/models/sequential/'''

model = tf.keras.Sequential() #Create the model
model.add(layers.Flatten(input_shape=(28, 28,1)))   # input layer: Flatten = to flatten I/p to 1-d: single list of i/p neuron corresponding to each pixel
model.add(layers.Dense(128, activation='relu'))     # one hidden layer
model.add(layers.Dense(128, activation='relu'))     # one hidden layer
model.add(layers.Dense(10, activation='softmax'))   # one output layer with 10 outputs (From 0 to 9)

model.summary() #to get the summary of the model
```
- Can extend from another pre-trained model, in this example is **MobileNet V2**.
```Python
# Create the base model from the pre-trained model MobileNet V2
IMG_SIZE = (minSize, minSize)
IMG_SHAPE = IMG_SIZE + (3,)
# instantiate a MobileNet V2 model pre-loaded with weights trained on ImageNet
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False, #Remove the final outut layer
                                               weights='imagenet')
base_model.trainable = False
model = tf.keras.Sequential(base_model)
#...continue add your own layers here like above example
model.add(layers.Dense(10, activation='softmax'))
```
#### 2.1.2. Method 2: using list of layers as input of `keras.Sequential()`
```Python
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

## 2.2. Functional API
- The functional API makes it easy to:
  - Manipulate multiple inputs and outputs. 
  - manipulate non-linear connectivity topologies -- these are models with layers that are not connected sequentially, which the Sequential API cannot handle.

```Python
#Layer construction using Functional API
input = tf.keras.Input(shape=(28,28,1), name="input_image")

x = layers.Conv2D(32, 3, activation='relu')(input)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPool2D(2)(x)
block_1_output = layers.Flatten()(x)

x = layers.Flatten()(input)
block_2_output = layers.Dense(64, activation='relu')(x)

x = layers.concatenate([block_1_output, block_2_output])
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(input, output, name="fashion_mnist") 

# To print the structure of the NN
tf.keras.utils.plot_model(model, "fashion_mnist.png", show_shapes=True)
```

## 2.3. Model Training
- Once the layers are constructed using either `Functional API` or `Sequential API`, we can compile the model
```Python
# Compile the model with appropriate Loss function. metrics is something you can monitor (but model does not optimize metrc)
# model.compile: to associate the NN with Loss Function
model.compile(optimizer=tf.optimizers.Adam(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
#interrupt training when it measures no progress on the validation set for a number of epochs (defined by the patience argument), 
# and it will optionally roll back to the best model
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,
                                                  restore_best_weights=True)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("best_model.h5",
                                                save_best_only=True)
                                                
# Run the stochastic gradient descent for specified epochs
history = model.fit(train_images, 
                    train_labels, 
                    validation_split=0.2, #using 20% of training data as validation
                    epochs=50,
                    batch_size=128,
                    callbacks=[checkpoint_cb, early_stopping_cb])
```
- `history` will contain
  - `history.history["accuracy"]`
  - `history.history["loss"]`
  - `history.epoch`

## CNN
- `keras.layers.GlobalMaxPooling2D()`: pool size = input size, usually used as last layer in the CNN before connecting to Dense Layers


## Resource
- [Keras Model - Pre-trained Models](https://keras.io/api/applications/)
- [TF Workshop](https://github.com/random-forests/tensorflow-workshop)
