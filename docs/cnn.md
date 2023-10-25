# CNN

## 1. CNN Basics

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

### 1.1. `SAME` vs `VALID` padding:

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

### 1.2. Important Layers in CNN

- `keras.layers.GlobalMaxPooling2D()`: pool size = input size, usually used as last layer in the CNN before connecting to Dense Layers
