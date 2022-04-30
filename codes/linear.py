import keras.layers
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
import tensorflow.keras.backend as k


class Linear(keras.layers.Layer):
    """定义全连接层"""

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias')
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        output = k.dot(inputs, self.kernel)
        if self.use_bias:
            output = k.bias_add(output, self.bias, data_format='channels_last')
        if self.activation == 'relu':
            output = keras.activations.relu(output)
        if self.activation == 'softmax':
            output = keras.activations.softmax(output)
        if self.activation == 'tanh':
            output = keras.activations.tanh(output)

        return output

