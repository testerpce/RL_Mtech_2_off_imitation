#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:04:49 2019

@author: sayambhu
"""

from collections import deque
import random
import numpy as np



class ReplayBuffer(object):
    
    def __init__(self,buffer_size,state_dim,action_dim,random_seed=123):
        """
        The right side of deque contains the most recent experiences
        """
        print("Creating Replay Buffer object")
        self.buffer_size=buffer_size
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.pointer=0
        self.states=np.zeros(shape=[buffer_size,state_dim])
        self.actions=np.zeros(shape=[buffer_size,action_dim])
        self.rewards=np.zeros(shape=[buffer_size,1])
        self.dones=np.zeros(shape=[buffer_size,1])
        self.next_states=np.zeros(shape=[buffer_size,state_dim])
        self.filled=False
        
        random.seed(random_seed)
        
    def add(self,s,a,r,t,s2):
        
        self.states[self.pointer] = s
        self.actions[self.pointer] = a
        self.rewards[self.pointer] = r
        self.dones[self.pointer] = t
        self.next_states[self.pointer] = s2
        
        self.pointer +=1
        if self.pointer%self.buffer_size == 0 :
            self.filled=True
        self.pointer =self.pointer%self.buffer_size
    
    def expert_add(self,s,a,r,t,s2):
        self.states[0:self.buffer_size]=s[0:self.buffer_size]
        self.actions[0:self.buffer_size]=a[0:self.buffer_size]
        self.rewards[0:self.buffer_size]=r[0:self.buffer_size]
        self.dones[0:self.buffer_size]=t[0:self.buffer_size]
        self.next_states[0:self.buffer_size]=s2[0:self.buffer_size]
        
        self.filled=True
        self.pointer=0
    
    def size(self):
        if self.filled:
            return self.buffer_size
        else:
            return self.pointer
    
    def sample_batch(self,batch_size):
        
        if self.filled :
# =============================================================================
#             print("reaching in filled")
# =============================================================================
            indexes=np.random.randint(low=0,high=self.buffer_size,size=batch_size)
            return self.states[indexes,:],self.actions[indexes,:],self.rewards[indexes,:], self.next_states[indexes, :]
        else:
# =============================================================================
#             print("reaching if not filled")
# =============================================================================
            if self.size() > batch_size:
# =============================================================================
#                 print("Reaching size greater condition")
# =============================================================================
                
                indexes=np.random.randint(low=0,high=self.pointer,size=batch_size)
            return self.states[indexes,:],self.actions[indexes,:],self.rewards[indexes,:], self.next_states[indexes, :]
        
        
        
    
    def clear(self):
        self.states=np.zeros(shape=[self.buffer_size,self.state_dim])
        self.actions=np.zeros(shape=[self.buffer_size,self.action_dim])
        self.rewards=np.zeros(shape=[self.buffer_size,1])
        self.dones=np.zeros(shape=[self.buffer_size,1])
        self.next_states=np.zeros(shape=[self.buffer_size,self.state_dim])
        self.pointer=0
        self.filled=False
        
    
        
        
        
        
        
        
    
    
        