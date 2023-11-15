# CTC Loss

- For example: a handwriting recognition model has images which are 32x128, & there are 32 time steps, and character list has a length of 80.

  - `input_length` is the length of the input sequence in time steps.
  - `label_length` is the length of the text label.
  - As 32 time steps, then `input_length=32` and your `label_length=12` (len("John Hancock")).
  <p align="center">
  <img src="../assets/img/ctc_loss_ex1.png" height="250"/>
  </p>

- Note: Processing input data in batches, which have to be padded to the largest element in the batch, so this information is lost. In handwritting case the `input_length` is always the **same**, but the `label_length` **varies**.
  - When dealing with speech recognition, for example, `input_length` can vary as well.

```Python
class CTCLayer(tf.keras.layers.Layer):
  def __init__(self, name=None):
      super().__init__(name=name)
      self.loss_fn = keras.backend.ctc_batch_cost # update by batch

  def call(self, y_true, y_pred):
      # Compute the training-time loss value and add it
      # to the layer using `self.add_loss()`.
      batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")

      input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
      label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

      input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
      label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

      loss = self.loss_fn(y_true, y_pred, input_length, label_length)
      self.add_loss(loss)

      # At test time, just return the computed predictions
      return y_pred

```

## Resources

- [An Intuitive Explanation of Connectionist Temporal Classification](https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c)
