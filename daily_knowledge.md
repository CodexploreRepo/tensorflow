# Daily Knowledge

## Day 1

- `tensorflow` vs GPU: if you have access to a GPU, TensorFlow will automatically use it whenever possible.
- `tensorflow-metal` initialization

  - Observation: When I run a script that uses keras or tensorflow with Apple M2, it shows the initialisation message

  ```Python
  2023-10-25 23:23:46.961496: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M2 Max
  2023-10-25 23:23:46.961512: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 32.00 GB
  2023-10-25 23:23:46.961517: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 10.67 GB
  2023-10-25 23:23:46.961547: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:306] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
  2023-10-25 23:23:46.961561: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:272] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
  ```

  - Surpress the print message
    - **Level 2** is to suppress these Warnings `(W)` and Informational `(I)` messages
    - **Level 3** is that All the messages (1 - informational `(I)`, 2 - warnings `(W)` and 3- errors`(E)`) will not be logged during code execution

  ```Python
  import os
  os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

  import tensorflow as tf
  ```
