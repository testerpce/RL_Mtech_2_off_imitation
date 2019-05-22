#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 23:57:35 2019

@author: sayambhu
"""
import tensorflow as tf
import numpy as np



class OrnsteinUhlenbeckActionNoise:
    def __init__(self,mu,sigma=0.3,theta=0.15,dt=1e-2,x0=None):
        self.theta=theta
        self.mu=mu
        self.sigma=sigma
        self.dt=dt
        self.x0=x0
        self.reset()
    
    def __call__(self):
        x=self.x_prev+self.theta*(self.mu-self.x_prev)*self.dt+self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev=x
        
        return x
    
    def reset(self):
        self.x_prev=self.x0 if self.x0 is not None else np.zeros_like(self.mu)
    
    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={},sigma={})'.format(self.mu,self.sigma)
    
#Tensorflow summary ops

def build_summaries():
    episode_reward=tf.Variable(0.)
    tf.summary.scalar('episode_reward',episode_reward)
    episode_ave_max_q=tf.Variable(0.)
    tf.summary.scalar('Qmax_value',episode_ave_max_q)
    summary_vars=[episode_reward,episode_ave_max_q]
    summary_ops=tf.summary.merge_all()
    
    return summary_ops,summary_vars

