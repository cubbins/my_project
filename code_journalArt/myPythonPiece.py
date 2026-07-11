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


class tempPiece():
    """
    Original external-attention LSTM class.

    Note: main_new.py does not currently use this class. It uses AttentionLSTM_t.
    This class was partly modernized, but it still depends on older LSTM internals
    such as step()/get_constants(), so prefer AttentionLSTM_t or a modern custom
    layer for new work.
    """

    def __init__(self):
        gg=1