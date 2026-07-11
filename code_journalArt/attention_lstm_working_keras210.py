#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
"""
Patched Keras 2.x version of the original attention_lstm.py.

Main repair:
- The old Keras-1 style attributes inner_init/output_dim/trainable_weights += ...
  were replaced in AttentionLSTM_t with modern self.add_weight(...), self.units,
  and self.recurrent_initializer.

This file keeps the original class names so main_new.py can still do:
    from attention_lstm import AttentionLSTM_t
"""

from keras import activations
from keras.layers import Wrapper
from keras.layers import InputSpec
from keras import backend as K
from keras.layers import LSTM


class AttentionLSTM(LSTM):
    """
    Original external-attention LSTM class.

    Note: main_new.py does not currently use this class. It uses AttentionLSTM_t.
    This class was partly modernized, but it still depends on older LSTM internals
    such as step()/get_constants(), so prefer AttentionLSTM_t or a modern custom
    layer for new work.
    """

    def __init__(self, output_dim, attention_vec, **kwargs):
        self.attention_vec = attention_vec
        super(AttentionLSTM, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        super(AttentionLSTM, self).build(input_shape)

        # Older Keras used _keras_shape; modern Keras often uses K.int_shape().
        attention_shape = K.int_shape(self.attention_vec)
        if attention_shape is None or len(attention_shape) < 2:
            raise ValueError("attention_vec must have shape (batch, attention_dim)")
        attention_dim = attention_shape[1]

        self.U_a = self.add_weight(
            name='{}_U_a'.format(self.name),
            shape=(self.units, self.units),
            initializer=self.recurrent_initializer,
            trainable=True,
        )
        self.b_a = self.add_weight(
            name='{}_b_a'.format(self.name),
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
        )

        self.U_m = self.add_weight(
            name='{}_U_m'.format(self.name),
            shape=(attention_dim, self.units),
            initializer=self.recurrent_initializer,
            trainable=True,
        )
        self.b_m = self.add_weight(
            name='{}_b_m'.format(self.name),
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
        )

        self.U_s = self.add_weight(
            name='{}_U_s'.format(self.name),
            shape=(self.units, self.units),
            initializer=self.recurrent_initializer,
            trainable=True,
        )
        self.b_s = self.add_weight(
            name='{}_b_s'.format(self.name),
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
        )

        # Keras 2 normally does not expose initial_weights on layers, but keep this
        # guard for old code that may set it manually before build().
        if getattr(self, 'initial_weights', None) is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        h, [h, c] = super(AttentionLSTM, self).step(x, states)
        attention = states[4]

        m = K.tanh(K.dot(h, self.U_a) + attention + self.b_a)
        s = K.exp(K.dot(m, self.U_s) + self.b_s)
        h = h * s

        return h, [h, c]

    def get_constants(self, x):
        constants = super(AttentionLSTM, self).get_constants(x)
        constants.append(K.dot(self.attention_vec, self.U_m) + self.b_m)
        return constants


class AttentionLSTM_t(LSTM):
    """
    Patched class used by main_new.py.

    Original code used old Keras-1 names:
        self.inner_init(...)
        self.output_dim
        self.trainable_weights += [...]

    Keras 2.x uses:
        self.add_weight(...)
        self.units
        automatic tracking of trainable weights
    """

    def __init__(self, output_dim, attn_activation='tanh', **kwargs):
        self.attn_activation = activations.get(attn_activation)
        super(AttentionLSTM_t, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        super(AttentionLSTM_t, self).build(input_shape)

        self.U_a = self.add_weight(
            name='{}_U_a'.format(self.name),
            shape=(self.units, self.units),
            initializer=self.recurrent_initializer,
            trainable=True,
        )

        self.b_a = self.add_weight(
            name='{}_b_a'.format(self.name),
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
        )

        self.U_s = self.add_weight(
            name='{}_U_s'.format(self.name),
            shape=(self.units, self.units),
            initializer=self.recurrent_initializer,
            trainable=True,
        )

        self.b_s = self.add_weight(
            name='{}_b_s'.format(self.name),
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
        )

        if getattr(self, 'initial_weights', None) is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        h, [h, c] = super(AttentionLSTM_t, self).step(x, states)

        m = K.tanh(K.dot(h, self.U_a) + self.b_a)
        alpha = K.exp(K.dot(m, self.U_s) + self.b_s)
        h = h * alpha

        return h, [h, c]

    def get_constants(self, x):
        constants = super(AttentionLSTM_t, self).get_constants(x)
        return constants


class AttentionLSTMWrapper(Wrapper):
    """
    Original wrapper class, modernized for weight creation.

    This class is not used by main_new.py, but the obvious old Keras symbols were
    also replaced here so the file no longer contains inner_init/output_dim usage.
    """

    def __init__(self, layer, attn_activation='tanh', single_attention_param=False, **kwargs):
        assert isinstance(layer, LSTM)
        self.supports_masking = True
        self.attn_activation = activations.get(attn_activation)
        self.single_attention_param = single_attention_param
        super(AttentionLSTMWrapper, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        super(AttentionLSTMWrapper, self).build(input_shape)

        units = self.layer.units

        self.U_a = self.add_weight(
            name='{}_U_a'.format(self.name),
            shape=(units, units),
            initializer=self.layer.recurrent_initializer,
            trainable=True,
        )
        self.b_a = self.add_weight(
            name='{}_b_a'.format(self.name),
            shape=(units,),
            initializer='zeros',
            trainable=True,
        )

        if self.single_attention_param:
            self.U_s = self.add_weight(
                name='{}_U_s'.format(self.name),
                shape=(units, 1),
                initializer=self.layer.recurrent_initializer,
                trainable=True,
            )
            self.b_s = self.add_weight(
                name='{}_b_s'.format(self.name),
                shape=(1,),
                initializer='zeros',
                trainable=True,
            )
        else:
            self.U_s = self.add_weight(
                name='{}_U_s'.format(self.name),
                shape=(units, units),
                initializer=self.layer.recurrent_initializer,
                trainable=True,
            )
            self.b_s = self.add_weight(
                name='{}_b_s'.format(self.name),
                shape=(units,),
                initializer='zeros',
                trainable=True,
            )

    def get_output_shape_for(self, input_shape):
        return self.layer.get_output_shape_for(input_shape)

    def step(self, x, states):
        h, [h, c] = self.layer.step(x, states)
        m = self.attn_activation(h)
        s = K.softmax(K.dot(m, self.U_s))
        if self.single_attention_param:
            h = h * K.repeat_elements(s, self.layer.units, axis=1)
        else:
            h = h * s

        return h, [h, c]

    def get_constants(self, x):
        constants = self.layer.get_constants(x)
        return constants

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        if K.backend() == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences. If your first layer is an '
                                'Embedding, make sure to pass it an '
                                '"input_length" argument.')
        if self.layer.stateful:
            initial_states = self.layer.states
        else:
            initial_states = self.layer.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.layer.preprocess_input(x)

        last_output, outputs, states = K.rnn(
            self.step,
            preprocessed_input,
            initial_states,
            go_backwards=self.layer.go_backwards,
            mask=mask,
            constants=constants,
            unroll=self.layer.unroll,
            input_length=input_shape[1],
        )
        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            return outputs
        return last_output
