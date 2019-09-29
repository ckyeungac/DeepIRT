import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from utils import getLogger

# set logger
logger = getLogger('Deep-IRT-model')


class MemoryHeadGroup():
    def __init__(self, memory_size, memory_state_dim, is_write, name="DKVMN-Head"):
        self.name = name 
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.is_write = is_write

    def correlation_weight(self, embedded_query_vector, key_memory_matrix):
        """
        Given a batch of queries, calculate the similarity between the query and 
        each key-memory slot via inner dot product. Then, calculate the weighting
        of each memory slot by softmax function. 

        Parameters:
            - embedded_query_vector (k): Shape (batch_size, key_memory_state_dim)
            - key_memory_matrix (D_k): Shape (memory_size, key_memory_state_dim)
        Result:
            - correlation_weight (w): Shape (batch_size, memory_size)
        """
        embedding_result = tf.matmul(
            embedded_query_vector, tf.transpose(key_memory_matrix)
        )
        correlation_weight = tf.nn.softmax(embedding_result)
        return correlation_weight

    def read(self, value_memory_matrix, correlation_weight):
        """
        Given the correlation_weight, read the value-memory in each memory slot
        by weighted sum. This operation is assumpted to be done in batch manner.

        Parameters:
            - value_memory_matrix (D_v): Shape (batch_size, memory_size, value_memory_state_dim)
            - correlation_weight (w): Shape (batch_size, memory_size)
        Result:
            - read_result (r): Shape (batch_size, value_memory_state_dim)
        """
        value_memory_matrix_reshaped = tf.reshape(value_memory_matrix, [-1, self.memory_state_dim])
        correlation_weight_reshaped = tf.reshape(correlation_weight, [-1,1])

        _read_result = tf.multiply(value_memory_matrix_reshaped, correlation_weight_reshaped) # row-wise multiplication
        read_result = tf.reshape(_read_result, [-1, self.memory_size, self.memory_state_dim])
        read_result = tf.reduce_sum(read_result, axis=1, keepdims=False)
        return read_result

    def write(self, value_memory_matrix, correlation_weight, embedded_content_vector, reuse=False):
        """
        Update the value_memory_matrix based on the correlation weight and embedded result vector.

        Parameters:
            - value_memory_matrix (D_v): Shape (batch_size, memory_size, value_memory_state_dim)
            - correlation_weight (w): Shape (batch_size, memory_size)
            - embedded_content_vector (v): Shape (batch_size, value_memory_state_dim)
            - reuse: indicate whether the weight should be reuse during training.
        Return:
            - new_value_memory_matrix: Shape (batch_size, memory_size, value_memory_state_dim)
        """
        assert self.is_write

        # erase_vector/erase_signal: Shape (batch_size, value_memory_state_dim)
        erase_signal = layers.fully_connected(
            inputs=embedded_content_vector, 
            num_outputs=self.memory_state_dim,
            scope=self.name+'/EraseOperation',
            reuse=reuse,
            activation_fn=tf.sigmoid
        )

        # add_vector/add_signal: Shape (batch_size, value_memory_state_dim)
        add_signal = layers.fully_connected(
            inputs=embedded_content_vector,
            num_outputs=self.memory_state_dim,
            scope=self.name+'/AddOperation',
            reuse=reuse,
            activation_fn=tf.tanh
        )

        # reshape from (batch_size, value_memory_state_dim) to (batch_size, 1, value_memory_state_dim)
        erase_reshaped = tf.reshape(erase_signal, [-1,1,self.memory_state_dim])
        # reshape from (batch_size, value_memory_state_dim) to (batch_size, 1, value_memory_state_dim)
        add_reshaped = tf.reshape(add_signal, [-1,1,self.memory_state_dim])
        # reshape from (batch_size, memory_size) to (batch_size, memory_size, 1)
        cw_reshaped = tf.reshape(correlation_weight, [-1, self.memory_size, 1])

        # erase_mul/add_mul: Shape (batch_size, memory_size, value_memory_state_dim)
        erase_mul = tf.multiply(erase_reshaped, cw_reshaped)
        add_mul = tf.multiply(add_reshaped, cw_reshaped)

        # Update value memory
        new_value_memory_matrix = value_memory_matrix * (1 - erase_mul)  # erase memory
        new_value_memory_matrix += add_mul

        return new_value_memory_matrix
        

class DKVMN():
    def __init__(self, memory_size, key_memory_state_dim, value_memory_state_dim,
                init_key_memory=None, init_value_memory=None, name="DKVMN"):
        self.name = name
        self.memory_size = memory_size
        self.key_memory_state_dim = key_memory_state_dim
        self.value_memory_state_dim = value_memory_state_dim

        self.key_head = MemoryHeadGroup(
            self.memory_size, self.key_memory_state_dim, 
            name=self.name+'-KeyHead', is_write=False
        )
        self.value_head = MemoryHeadGroup(
            self.memory_size, self.value_memory_state_dim, 
            name=self.name+'-ValueHead', is_write=True
        )

        self.key_memory_matrix = init_key_memory
        self.value_memory_matrix = init_value_memory

    def attention(self, embedded_query_vector):
        correlation_weight = self.key_head.correlation_weight(
            embedded_query_vector=embedded_query_vector, 
            key_memory_matrix=self.key_memory_matrix
        )
        return correlation_weight

    def read(self, correlation_weight):
        read_content = self.value_head.read(
            value_memory_matrix=self.value_memory_matrix, 
            correlation_weight=correlation_weight
        )
        return read_content

    def write(self, correlation_weight, embedded_result_vector, reuse):
        self.value_memory_matrix = self.value_head.write(
            value_memory_matrix=self.value_memory_matrix, 
            correlation_weight=correlation_weight, 
            embedded_content_vector=embedded_result_vector, 
            reuse=reuse
        )
        return self.value_memory_matrix