# Deep Learning

## Tensorflow Modules
```Python
#Tensorflow
import tensorflow as tf

tf.feature_column. #Feature Columns
```
- Keras 
```Python
#Keras
from tensorflow import keras

#Datasets
keras.datasets.mnist.load_data()

#Model Creation 
keras.Input(shape=(32, 32, 3)) #Input shape
keras.layers. #Keras Layers
keras.models. #Keras Models

#Utils
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
```

## CNN
- `keras.layers.GlobalMaxPooling2D()`: pool size = input size, usually used as last layer in the CNN before connecting to Dense Layers


## Resource
- [Keras Model - Pre-trained Models](https://keras.io/api/applications/)
- [TF Workshop](https://github.com/random-forests/tensorflow-workshop)
