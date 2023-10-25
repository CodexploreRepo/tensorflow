# Introduction to Keras and Tensorflow

## 1. What is Tensorflow

- TensorFlow is a Python-based, free, open source machine learning platform, developed primarily by Google.
- It’s really a **PLATFORM**, home to a vast ecosystem of components, some developed by Google and some developed by third parties.
- For instance:
  - `TF-Agents` for reinforcement-learning research
  - `TFX` for industry-strength machine learning workflow management
  - `TensorFlow Serving` for production deployment
  - `TensorFlow Hub` repository of pretrained models.

## 2. What is Keras

- Keras is a deep learning API for Python, built on top of TensorFlow, that provides a convenient way to define and train any kind of deep learning model.
- Through TensorFlow, Keras can run on top of different types of hardware: GPU, TPU, or plain CPU—and can be seamlessly scaled to thousands of machines.
- TensorFlow is a low-level tensor computing platform, and Keras is a high-level deep learning API
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/223384829-37802e97-8a1a-423d-9eed-868b08995864.png" height="250"/>
</p>

## 3. Common Concepts

### Low-level tensor manipulation

- Low-level tensor manipulation—the infrastructure that underlies all modern machine learning.
- This translates to **TensorFlow APIs**:
  - _Tensors_: including special tensors that store the network’s state (variables)
  - _Tensor operations_ such as `addition`, `relu`, `matmul`
  - _Backpropagation_ a way to compute the gradient of mathematical expressions (handled in TensorFlow via the `GradientTape` object)

### High-level deep learning concepts

- This translates to Keras APIs:
  - _Layers_, which are combined into a model
  - _Loss function_ which defines the feedback signal used for learning
  - _Optimizer_ which determines how learning proceeds
  - _Metrics_ to evaluate model performance, such as accuracy
  - _Training loop_ that performs mini-batch stochastic gradient descent
  - _Metrics_ to evaluate model performance, such as accuracy
