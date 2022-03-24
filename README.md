# Deep Learning

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Tensorflow Modules](#1-tensorflow-modules)
  - [1.1. Keras](#11-keras)  
- [2. Model Creation](#2-model-creation)
  - [2.1. Sequential API](#21-sequential-api)
  - [2.2. Functional API](#22-functional-api)
  - [2.3. Model Training](#23-model-training) 
  - [2.4. Model Evaluation](#24-model-evaluation)  
- [3. CNN](#3-cnn)
  - [3.1. CNN Basics](#31-cnn-basics) 
- [Resources](#resources)

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
keras.Input(shape=(32, 32, 3)) #Input for Functional API
keras.layers. #Keras Layers


#Utils
#To plot the model structure 
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
```

[(Back to top)](#table-of-contents)

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

tf.keras.utils.plot_model(model, "fashion_mnist.png", show_shapes=True) # To print the structure of the NN
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
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])
```

[(Back to top)](#table-of-contents)
## 2.2. Functional API
- The functional API makes it easy to:
  - Manipulate multiple inputs and outputs. 
  - Manipulate non-linear connectivity topologies -- these are models with layers that are not connected sequentially, which the Sequential API cannot handle.

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
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/158103268-7a0813b1-300d-44ff-a606-370baa3c58d9.png" width="800" height="700" />
</p>


[(Back to top)](#table-of-contents)
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
  - `history.history["accuracy"]`: a list of accuracy per epoch for both train & val (`"val_accuracy"`) set
  - `history.history["loss"]`: a list of loss per epoch for both train & val (`"val_loss"`) set
  - `history.epoch`: number of epochs run
- To visualize **history** information

```Python
def plot_history(ax, key):
    val = ax.plot(history.epoch, history.history['val_'+key], '--', label='validation_' + key)
    ax.plot(history.epoch, history.history[key], color=val[0].get_color(),label= 'training_' + key)
    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel(key)
    ax.set_xlim([0,max(history.epoch)])

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols=2, figsize=(15,6))
plot_history(ax1, 'accuracy')
plot_history(ax2, 'loss')
plt.show()
```

[(Back to top)](#table-of-contents)
## 2.4. Model Evaluation
- Evaluate the model on the test set using `model.evaluate()`
```Python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', round(test_acc,3))
```
- Compute confusion matrix using `model.predict()`
```Python
pred_proba = model.predict(test_images) #.predict() return the prob of each class
predictions = np.argmax(pred_proba, axis = 1)  #to convert from prob to class number using argmax
print("Confusion Matrix: ")
pd.DataFrame(confusion_matrix(test_labels, predictions), index=[f'actual_{i}' for i in range(10)], columns=[f'pred_{i}' for i in range(10)])
```
- Visualize the prediction vs actual label
```Python
# Code to visualize predictions
# Correct predictions are highlighted in green
# Incorrect predictions are highlighted in red
import numpy as np
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i][:,:,0], cmap=plt.cm.binary)
    predicted_label = predictions[i]
    true_label = test_labels[i]
    if predicted_label == true_label:
      color = 'green'
    else:
      color = 'red'
    plt.xlabel("{} ({})".format(label_names[predicted_label], 
                                label_names[true_label]),
                                color=color)
```

[(Back to top)](#table-of-contents)
# 3. CNN
## 3.1. CNN Basics
```Python
#Convolution Layer
x = layers.Conv2D(filters=5,
                  kernel_size=(3,3),
                  strides = 2,
                  padding = 'SAME',
                  activation='relu')(x)
#simplified param input for Conv2D: filters=64, kernel_size=(3x3)         
x = layers.Conv2D(64, 3, activation='relu')(x)

#Pooling Layer
x = layers.MaxPool2D(pool_size=(2, 2), 
                     strides=2,
                     padding = 'SAME')(x)
#simplified param input for Pool: pool_size = 2x2
x = layers.MaxPool2D(2)(x)
```
### 3.1.1. `SAME` vs `VALID` padding:
- Method 1: `VALID` padding means use no padding ("assume" that all dimensions are valid so that input image fully gets covered by filter and stride you specified)
```Python
#How to compute output size with `VALID` padding
out_height = ceil((in_height - filter_height + 1) / stride_height)
out_width  = ceil((in_width - filter_width + 1) / stride_width)
```
- Method 2: `SAME` padding is applied to each spatial dimension. When the strides are 1, the input is padded such that the output size is the same as the input size.
```Python
#How to compute output size with `SAME` padding
out_height = ceil(in_height / stride_height)
out_width  = ceil(in_width / stride_width)
```

### 3.1.2. Important Layers in CNN
- `keras.layers.GlobalMaxPooling2D()`: pool size = input size, usually used as last layer in the CNN before connecting to Dense Layers

[(Back to top)](#table-of-contents)

# Resources
- [Keras Model - Pre-trained Models](https://keras.io/api/applications/)
- [TF Workshop](https://github.com/random-forests/tensorflow-workshop)
- [Generate Images with BigGAN](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/biggan_generation_with_tf_hub.ipynb)
## Todo List
- [Autoencoders in Practice: Dimensionality Reduction and Image Denoising](https://towardsdatascience.com/autoencoders-in-practice-dimensionality-reduction-and-image-denoising-ed9b9201e7e1)
