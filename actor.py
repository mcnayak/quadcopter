from keras.models import Model
from keras.layers import Input, Dense, Add, Lambda
from keras.optimizers import Adam
import keras.backend as KB
import copy
import numpy as np


class Actor():
    def __init__(self, state_size, action_size, action_low, action_high):

        self.state_size = state_size
        self.action_size = action_size

        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        self.build_actor(self.state_size, self.action_size)
        
    def build_actor(self, state_size, action_size):
        """ Build an actor(policy) nework that maps states -> actions."""

        l1_size = 32
        l2_size = 64
        l3_size = 32
        
        #Add hidden layers
        states = Input(shape=[state_size], name='states')
        l1 = Dense(l1_size, activation='relu', name='hidden1')(states)
        l2 = Dense(l2_size, activation='relu', name='hidden2')(l1)
        l3 = Dense(l3_size, activation='relu', name='hidden3')(l2)
        
        raw_actions = Dense(action_size, activation='sigmoid', name='raw_actions')(l3)
        
        actions = Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)
        print(actions)
        
        #Create Keras model
        self.model = Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = Input(shape=([self.action_size]), name='action_grads')
        loss = KB.mean(-action_gradients * actions)
        optimizer = Adam(lr=0.001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = KB.function(
            inputs=[self.model.input, action_gradients, KB.learning_phase()],
            outputs=[],
            updates=updates_op)