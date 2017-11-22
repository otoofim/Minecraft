# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network
class Network(nn.Module):

    #Iitialising neural network
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, 30)

        #-- Not used as this is designed for implementing an extra dimension, but we only require 2D here
        #self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30, nb_action)
    
    #To increase and decrease number of layers, it can be altered by commenting or uncommentign lines below
    def forward(self, state, tra=True):
        x1 = F.relu(self.fc1(state))
        x2 = F.relu(self.fc2(x1))

        #-- Not used as these are designed for implementing an extra dimension, but we only require 2D here
        #x3 = F.dropout(F.relu(self.fc3(x2)), p = 0.4, training = tra)
        #x3 = F.relu(self.fc3(x2))
        q_values = self.fc4(x2)
        return q_values

# Implementing Experience Replay
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning
class Dqn():

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    #Selecting an action for a state
    def select_action(self, state):
        # In tensors, computational graph and gradients are included. However, it is not required to use them to compute probability distributions with softmax. To prevent computational graph and gradients to be included Volatile is set to "True".
        probs = F.softmax(self.model(Variable(state, volatile = True), False)*100) # Default calibrated to T=100
        
        #epsilon greedy strategy
        x = random.uniform(0, 1)
        if x < 0.9:
            _, temp = torch.max(probs[0], 0)
            action = temp.data[0]
        else:

            action = random.randint(0, 3)

        #action = probs.multinomial().data[0,0]
        return action

    #Implementing learning process
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()
    #Updating the network based on new state and reward
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 1000:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(1000)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    #Returning ave of rewards in a episod.
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    #Saves the neural network weights.
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    #Loading current saved neural network.
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
