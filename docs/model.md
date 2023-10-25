# Tensorflow Model

## 1. Model Creation

### 1.1. Sequential API

- There are 2 ways to create a model using Sequential API

#### 1.1.1. Method 1: using `add` method

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

#### 1.1.2. Method 2: using list of layers as input of `keras.Sequential()`

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

### 1.2. Functional API

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

## 2. Model Compilation

- Once the layers are constructed using either `Functional API` or `Sequential API`, we can compile the model

```Python
# Shortcut: optimizer, loss, and metrics as strings
model.compile(optimizer="rmsprop",
              loss="mean_squared_error",
              metrics=["accuracy"])
# Python object: optimizer, loss, and metrics
model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.BinaryAccuracy()])
# Custom metrics
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
              loss=my_custom_loss,
              metrics=[my_custom_metric_1, my_custom_metric_2])
```

<details>
  <summary>Optimizer</summary>
  
  * SGD (with or without momentum)
  * RMSprop
  * Adam
  * Adagrad
</details>

<details>
  <summary>Loss</summary>
  
  * CategoricalCrossentropy
  * SparseCategoricalCrossentropy
  * BinaryCrossentropy
  * MeanSquaredError
  * KLDivergence
  * CosineSimilarity
</details>

<details>
  <summary>Metrics</summary>
  
  * CategoricalAccuracy
  * SparseCategoricalAccuracy
  * BinaryAccuracy
  * AUC
  * Precision
  * Recall
</details>

## 3. Model Training

- `fit()` method implements the training loop itself. These are its key arguments:
  - The **data** (inputs and targets) to train on. `NumPy arrays` or a `TensorFlow Dataset` object
  - The **number of epochs** to train for: how many times the training loop should iterate over the data passed.
  - The **batch size** to use within each epoch of mini-batch gradient descent: the number of training examples considered to compute the gradients for one weight update step.

```Python
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

## 4. Model Evaluation

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

## 5. Model Inference

```Python
# Method 1: using __call__() method of model
# Input: a NumPy array or TensorFlow tensor -> a TensorFlow tensor
# Drawback: his will process all inputs in new_inputs at once, which may not be feasible if you’re looking at a lot of data (in particular, it may require more memory than your GPU has).
predictions = model(new_inputs)

# Method 2: model.predict() -> iterate over the data in small batches and return a NumPy array of predictions.
# Input: NumPy array or a Dataset -> a NumPy array
predictions = model.predict(new_inputs, batch_size=128)
```
