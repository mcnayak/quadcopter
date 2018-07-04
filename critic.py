from keras.models import Model
from keras.layers import Input, Dense, Add, Lambda
from keras.optimizers import Adam
import keras.backend as K
import copy
import numpy as np
import task as Task

class Critic:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.build_critic()

    def build_critic(self,learning_rate=0.05):
        l1 = 32
        l2 = 64
        l3 = 16
         
        # Define input layers for state pathway
        states = Input(shape=[self.state_size])
        actions = Input(shape=[self.action_size])
        
        s_1 = Dense(l1, activation='relu')(states)
        s_2 = Dense(l2, activation='relu')(s_1)

        # Add hidden layers(s) for action pathway
        
        a_1 = Dense(l1, activation='relu')(actions)
        a_2 = Dense(l2, activation='relu')(a_1)

        # Try different layer sizes, activations, add batch normalization, regulaizers, etc.
        # Combine state and action pathways
        net = Add()([s_2, a_2])
        net = Dense(l3, activation='relu')(net)

        
        # Add final output layer to prduce action values (Q values)
        Q_values = Dense(units=1, name='q_values')(net)
    
        # Create Keras Model
        self.model = Model(inputs=[states, actions], outputs=Q_values)

        optimizer = Adam()
        self.model.compile(loss='mse', optimizer=optimizer)

        action_gradients = K.gradients(Q_values, actions)
        
        # Define an additional function to fetch action gradients ( to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)



class OUNoise:
    def __init__(self, size, mu, theta, sigma):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def add_noise(self, action):
        off = np.random.random(1)[0]
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        action = action + self.state
        if off > 0.999:
            action *= 0
        return action
    
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state