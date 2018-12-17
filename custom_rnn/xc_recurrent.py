#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@software: PyCharm Community Edition
@file: xc_recurrent.py
@time: 12/17/18 1:48 PM
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings
from keras import backend as K
from keras.layers import activations, initializers, regularizers, constraints
from keras.engine import Layer, InputSpec
from keras.utils.generic_utils import  has_arg
from keras.legacy.layers import Recurrent
from keras.legacy import interfaces


class StackedRNNCells(Layer):

    def __init__(self, cells, **kwargs):

        for cell in cells:
            if not hasattr(cell, 'call'):
                raise ValueError('All cells must have a `call` method. '
                                 'received cells:', cells)
            if not hasattr(cell, 'state_size'):
                raise ValueError('All cells must have a '
                                 '`state_size` attribute. '
                                 'received cells:', cells)

        self.cells = cells
        super(StackedRNNCells, self).__init__(**kwargs)

    @property
    def state_size(self):
        state_size = []
        for cell in self.cells:
            if hasattr(cell.state_size, '__len__'):
                state_size += list(cell.state_size)
            else:
                state_size.append(cell.state_size)
        return tuple(state_size)

    def call(self, inputs, states, constants=None, **kwargs):
        nested_states = []
        for cell in self.cells[::-1]:
            if hasattr(cell.state_size, '__len__'):
                nested_states.append(states[:len(cell.state_size)])
                states = states[len(cell.state_size):]
            else:
                nested_states.append([states[0]])
                states = states[1:]
        nested_states = nested_states[::-1]
        new_nested_states = []

        for cell, states in zip(self.cells, nested_states):
            if has_arg(cell.call, 'constants'):
                inputs, states = cell.call(inputs, states,
                                           constraints=constraints,
                                           **kwargs)
            else:
                inputs, states = cell.call(inputs, states, **kwargs)

            new_nested_states.append(states)

        states = []
        for cell_states in nested_states[::-1]:
            states += cell_states
        return inputs, states

    def build(self, input_shape):
        if isinstance(input_shape, list):
            constraints_shape = input_shape[1:]
            input_shape = input_shape[0]
        for cell in self.cells:
            if isinstance(cell, Layer):
                if hasattr(cell.call, 'constants'):
                    cell.build([input_shape] + constraints_shape)
                else:
                    cell.build(input_shape)
            if hasattr(cell.state_size, '__len__'):
                output_dim = cell.state_size[0]
            else:
                output_dim = cell.state_size
            input_shape = (input_shape[0], output_dim)
        self.built = True

    def get_config(self):
        cells = []
        for cell in self.cells:
            cells.append({'class_name': cell.__class__.__name__,
                          'config': cell.get_config()})
        config = {'cells': cells}
        base_config = super(StackedRNNCells, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from keras.layers import deserialize as deserialize_layer
        cells = []
        for cell_config in config.pop('cells'):
            cells.append(deserialize_layer(cell_config,
                                           custom_objects=custom_objects))
        return cls(cells, **config)

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        weights = []
        for cell in self.cells:
            if isinstance(cell, Layer):
                weights = cell.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = []
        for cell in self.cells:
            if isinstance(cell, Layer):
                weights += cell.non_trainable_weights
        if not self.trainable:
            trainable_weights = []
            for cell in self.cells:
                if isinstance(cell, Layer):
                    trainable_weights += cell.trainable_weights
            return trainable_weights + weights
        return weights

    def get_weights(self):
        weights = []
        for cell in self.cells:
            if isinstance(cell, Layer):
                weights += cell.weights
        return K.batch_get_value(weights)

    def set_weights(self, weights):
        tuples = []
        for cell in self.cells:
            if isinstance(cell, Layer):
                num_param = len(cell.weights)
                weights = weights[:num_param]
                for sw, w in zip(cell.weights, weights):
                    tuples.append((sw, w))
                weights = weights[num_param:]
        K.batch_get_value(tuples)

    @property
    def losses(self):
        losses = []
        for cell in self.cells:
            if isinstance(cell, Layer):
                cell_losses = cell.losses
                losses += cell_losses
        return losses

    def get_losses_for(self, inputs):
        losses = []
        for cell in self.cells:
            if isinstance(cell, Layer):
                cell_losses = cell.get_losses_for(inputs)
                losses += cell_losses
        return losses


class RNN(Layer):

    def __init__(self, cell,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if isinstance(cell, (list, tuple)):
            cell = StackedRNNCells(cell)
        if not hasattr(cell, 'call'):
            raise ValueError('`cell` should have a `call` method. '
                             'The RNN was passed:', cell)
        if not hasattr(cell, 'state_size'):
            raise ValueError('The RNN cell should have '
                             'an attribute `state_size` '
                             '(tuple of integers, '
                             'one integer per RNN state).')
        super(RNN, self).__init__(**kwargs)
        self.cell = cell
        self.return_sequence = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll

        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        self.state_spec = None
        self._states = None
        self.constants_spec = None
        self._num_constants = None

    @property
    def states(self):
        if self._states is None:
            if isinstance(self.cell.state_size, int):
                num_states = 1
            else:
                num_states = len(self.cell.state_size)
            return [None for _ in range(num_states)]

    @states.setter
    def states(self, states):
        self._states = states

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if hasattr(self.cell.state_size, '__len__'):
            state_size = self.cell.state_size
        else:
            state_size = [self.cell.state_size]
        output_dim = state_size[0]

        if self.return_sequence:
            output_shape = (input_shape[0], input_shape[1], output_dim)
        else:
            output_shape = (input_shape[0], output_dim)

        if self.return_state:
            state_shape = [(input_shape[0], dim) for dim in state_size]
            return [output_shape] + state_shape
        else:
            return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        output_mask = mask if self.return_sequence else None
        if self.return_state:
            state_mask = [None for _ in self.states]
            return [output_mask] + state_mask
        else:
            return output_mask

    def build(self, input_shape):
        if self._num_constants is not None:
            constraints_shape = input_shape[-self._num_constants:]
        else:
            constraints_shape = None

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size= input_shape[0] if self.stateful else None
        input_dim = input_shape[-1]
        self.input_spec[0] = InputSpec(shape=(batch_size, None, input_dim))

        if isinstance(self.cell, Layer):
            step_input_shape = (input_shape[0], ) + input_shape[2:]
            if constraints_shape is not None:
                self.cell.build([step_input_shape] + constraints_shape)
            else:
                self.cell.build(step_input_shape)

        if hasattr(self.cell.state_size, '__len__'):
            state_size = list(self.cell.state_size)
        else:
            state_size = [self.cell.state_size]

        if self.state_spec is not None:
            if [spec.shape[-1] for spec in self.state_spec] != state_size:
                raise ValueError(
                    'An `initial_state` was passed that is not compatible with '
                    '`cell.state_size`. Received `state_spec`={}; '
                    'however `cell.state_size` is '
                    '{}'.format(self.state_spec, self.cell.state_size)
                )
            else:
                self.state_spec = [InputSpec(shape=(None, dim)) for dim in state_size]

        if self.stateful:
            self.reset_states()

    def get_initial_state(self, inputs):
        initial_state = K.zeros_like(inputs)
        initial_state = K.sum(initial_state, axis=(1, 2))
        initial_state = K.expand_dims(initial_state)

        if hasattr(self.cell.state_size, '__len__'):
            return [K.tile(initial_state, [1, dim])
                    for dim in self.cell.state_size]
        else:
            return [K.tile(initial_state, [1, self.cell.state_size])]

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        inputs, initial_state, constants = self._standardize_args(
            inputs, initial_state, constants)

        if initial_state is None and constraints is None:
            return sum(RNN, self).__call__(inputs, **kwargs)

        additional_inputs = []
        additional_specs = []

        if initial_state is not None:
            kwargs['inital_state'] = initial_state
            additional_inputs += initial_state
            self.state_spec = [InputSpec(shape=K.int_shape(state))
                               for state in initial_state]
            additional_specs += self.state_spec
        if constraints is not None:
            kwargs['constants'] = constants
            additional_inputs += constants
            self.constants_spec = [InputSpec(shape=K.int_shape(constant))
                                   for constant in constants]
            self._num_constants = len(constants)
            additional_specs += self.constants_spec
            # at this point additional_inputs cannot be empty
            is_keras_tensor = K.is_keras_tensor(additional_inputs[0])
            for tensor in additional_inputs:
                if K.is_keras_tensor(tensor) != is_keras_tensor:
                    raise ValueError('The initial state or constants of an RNN'
                                     ' layer cannot be specified with a mix of'
                                     ' Keras tensors and non-Keras tensors'
                                     ' (a "Keras tensor" is a tensor that was'
                                     ' returned by a Keras layer, or by `Input`)')

            if is_keras_tensor:
                # Compute the full input spec, including state and constants
                full_input = [inputs] + additional_inputs
                full_input_spec = self.input_spec + additional_specs
                # Perform the call with temporarily replaced input_spec
                original_input_spec = self.input_spec
                self.input_spec = full_input_spec
                output = super(RNN, self).__call__(full_input, **kwargs)
                self.input_spec = original_input_spec
                return output
            else:
                return super(RNN, self).__call__(inputs, **kwargs)







