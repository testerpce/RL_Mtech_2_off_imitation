#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 23:49:44 2019

@author: sayambhu
"""

import tensorflow as tf
import numpy as np
import tflearn


class ActorNetwork(object):
    """
    Using almost same actor as ddpg except I am keeping batch normalization
    to be an option
    """
    
    def __init__(self,sess, state_dim,action_dim,action_bound,learning_rate,tau,batch_size,actor_bn=True):
        self.sess=sess
        self.s_dim=state_dim
        self.a_dim=action_dim
        self.action_bound=action_bound
        self.learning_rate=learning_rate
        self.tau=tau
        self.batch_size=batch_size
        self.actor_bn=actor_bn
        
        #Actor Network
        self.inputs,self.out,self.scaled_out=self.create_actor_network(reuse=False,original=True)
        
        self.network_params=tf.trainable_variables("actor")
        
        #Target Network
        self.target_inputs,self.target_out,self.target_scaled_out=self.create_actor_network(reuse=False,original=False)
        self.target_network_params=tf.trainable_variables("target_a")#[len(self.network_params):]
        
# =============================================================================
#         print("actor params = ",self.network_params,"target actor params = ",self.target_network_params)
# =============================================================================
        
        #Op for periodically updating target network with online network
        self.update_target_network_params=[self.target_network_params[i].assign(tf.multiply(self.network_params[i],self.tau)+tf.multiply(self.target_network_params[i],1.-self.tau)) for i in range(len(self.target_network_params))]
        
        #This gradient will be provided by critic network
        self.action_gradient=tf.placeholder(tf.float32,[None,self.a_dim])
        
        #Combine gradients here
        self.unnormalized_actor_gradients=tf.gradients(self.scaled_out,self.network_params,-self.action_gradient)
# =============================================================================
#         print(self.unnormalized_actor_gradients)
# =============================================================================
        self.actor_gradients=list(map(lambda x: tf.div(x,self.batch_size),self.unnormalized_actor_gradients))
        
        #Optimization op
        self.optimize=tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients,self.network_params))
        
        self.num_trainable_vars=len(self.network_params)+len(self.target_network_params)
    
    
    def create_actor_network(self,reuse=False,original=True):
        
        inputs=tflearn.input_data(shape=[None,self.s_dim])
        
        if original:
            ns="actor"
        else:
            ns="target_a"
            
        with tf.variable_scope(ns) as scope:
            
            if reuse:
                tf.get_variable_scope().reuse_variables()
            
            if self.actor_bn:
                net=tflearn.fully_connected(inputs,400,name="FC_1")
                net=tflearn.layers.normalization.batch_normalization(net)
                net=tflearn.activations.relu(net)
                net=tflearn.fully_connected(net,300,name="FC_2")
                net=tflearn.layers.normalization.batch_normalization(net)
                net=tflearn.activations.relu(net)
                
            else:
                net=tflearn.fully_connected(inputs,64,name="FC_1")
                net=tflearn.activations.relu(net)
                net=tflearn.fully_connected(net,64,name="FC_2")
                net=tflearn.activations.relu(net)
                
            #Final layer weights between -0.003 to 0.003
            w_init=tflearn.initializations.uniform(minval=-0.003,maxval=0.003)
            out=tflearn.fully_connected(net,self.a_dim,activation='tanh',weights_init=w_init)
            
            scaled_out=tf.multiply(out,self.action_bound)
        return inputs,out,scaled_out
    
    def train(self,inputs,a_gradient):
        return self.sess.run(self.optimize,feed_dict={self.inputs:inputs,self.action_gradient:a_gradient})
        
    def predict(self,inputs):
        return self.sess.run(self.scaled_out,feed_dict={self.inputs:inputs})
    
    def predict_target(self,inputs):
        return self.sess.run(self.target_scaled_out,feed_dict={self.target_inputs:inputs})
    
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
    
    def get_num_trainable_vars(self):
        return self.num_trainable_vars

class CriticNetwork(object):
    """
    Same as ddpg but with optional batch normalization 
    and a separate loss function
    """
    
    def __init__(self,sess,state_dim,action_dim,learning_rate,tau,gamma,num_actor_vars,critic_bn=True):
        self.sess=sess
        self.s_dim=state_dim
        self.a_dim=action_dim
        self.learning_rate=learning_rate
        self.tau=tau
        self.gamma=gamma
        self.critic_bn=critic_bn
        
# =============================================================================
#         self.expert_inputs=tflearn.input_data(shape=[None,self.s_dim])
#         self.expert_action=tflearn.input_data(shape=[None,self.a_dim])
# =============================================================================
        
        #create the critic network
        self.inputs,self.action,self.out=self.create_critic_network(reuse=False,original=True)
        self.network_params=tf.trainable_variables("critic")#[num_actor_vars:]
        
        #Target network
        self.target_inputs,self.target_action,self.target_out=self.create_critic_network(reuse=False,original=False)
        self.target_network_params=tf.trainable_variables("target_c")#[(len(self.network_params)+num_actor_vars):]
        
        #Expert network params
        self.expert_inputs,self.expert_action,self.expert_out=self.create_critic_network(reuse=True,original=True)
        self.expert_network_params=tf.trainable_variables("critic")#[(len(self.network_params)+num_actor_vars+len(self.target_network_params)):]
        
# =============================================================================
#         print("actor params = ",num_actor_vars,"critic params =",self.network_params,"expert params =",self.expert_network_params,"target params = ",self.target_network_params,"total params = ",tf.trainable_variables())
# =============================================================================
        
        #Op for periodically updating target network with online network
        self.update_target_network_params=[self.target_network_params[i].assign(tf.multiply(self.network_params[i],self.tau)+tf.multiply(self.target_network_params[i],1.-self.tau)) for i in range(len(self.target_network_params))]
# =============================================================================
#         self.update_network_to_expert_params=[self.expert_network_params[i].assign(self.network_params[i]) for i in range(len(self.expert_network_params))]
#         self.update_expert_to_network_params=[self.network_params[i].assign(self.expert_network_params[i]) for i in range(len(self.network_params))]
# 
# =============================================================================
        
        #Network target y[i]
        self.predicted_q_value=tf.placeholder(tf.float32,[None,1])
        self.expert_q_value=tf.placeholder(tf.float32,[None,1])
        
        #Define loss and optimization op
        self.loss=self.JS_loss(self.expert_q_value,self.expert_out,self.predicted_q_value,self.out,self.gamma)
        self.optimize=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        #Gradient of output (i.e. Q value) is taken with respect to the action
        self.action_grads=tf.gradients(self.out,self.action)
    
    def JS_loss_expert(self,predicted_q_value,out,gamma):
        
        predicted_q_value=tf.math.pow(predicted_q_value,gamma)
        def KL_loss(p,q):
            kl=p*tf.math.log(p+1e-12)-p*tf.math.log(q+1e-12)+(1.-p)*tf.math.log(1.-p+1e-12)-(1.-p)*tf.math.log(1.-q+1e-12)
            
            return kl
        M=(predicted_q_value+out)/2
        L1=KL_loss(out,M)
        L2=KL_loss(predicted_q_value,M)
        JSD_E=0.5*(L1+L2)
        
        return JSD_E
    
    def JS_loss_non_expert(self,predicted_q_value,out,gamma):
        
        predicted_q_value=tf.math.pow(predicted_q_value,gamma)/2
        def KL_loss(p,q):
            kl=p*tf.math.log(p+1e-12)-p*tf.math.log(q+1e-12)+(1.-p)*tf.math.log(1.-p+1e-12)-(1.-p)*tf.math.log(1.-q+1e-12)
            
            return kl
        M=(predicted_q_value+out)/2
        L1=KL_loss(out,M)
        L2=KL_loss(predicted_q_value,M)
        JSD_NE=0.5*(L1+L2)
        
        return JSD_NE
    
    def JS_loss(self,expert_predicted_q_value,expert_out,
                predicted_q_value,out,gamma):
        
        return self.JS_loss_expert(expert_predicted_q_value,
                                   expert_out,
                                   gamma)+self.JS_loss_non_expert(predicted_q_value,
                                        out,gamma)
    
    def create_critic_network(self,reuse=False,original=True):
        inputs=tflearn.input_data(shape=[None,self.s_dim])
        action=tflearn.input_data(shape=[None,self.a_dim])
        
        if original:
            ns="critic"
        else:
            ns="target_c"
            
        with tf.variable_scope(ns) as scope:
            
            if reuse:
                tf.get_variable_scope().reuse_variables()
            
            if self.critic_bn:
                net=tflearn.fully_connected(inputs,400)
                net=tflearn.layers.normalization.batch_normalization(net)
                net=tflearn.activations.relu(net)
                
                #Adding action in second hidden layer
                t1=tflearn.fully_connected(net,300)
                t2=tflearn.fully_connected(action,300)
                net=tflearn.activation(tf.matmul(net,t1.W)+tf.matmul(action,t2.W)+t2.b,activation='relu')
                
                
            else:
                net=tflearn.fully_connected(inputs,64)
                net=tflearn.activations.relu(net)
                t1=tflearn.fully_connected(net,64)
                t2=tflearn.fully_connected(action,64)
                net=tflearn.activation(tf.matmul(net,t1.W)+tf.matmul(action,t2.W)+t2.b,activation='relu')
            
            #linear layer connected to one output representing Q(s,a)
            w_init=tflearn.initializations.uniform(minval=-0.003,maxval=0.003)
            out=tflearn.fully_connected(net,1,weights_init=w_init,activation='sigmoid')
            
        return inputs,action,out
    
    def train(self,expert_inputs,expert_actions,inputs,action,predicted_q_value,expert_q_value):
        return self.sess.run([self.out,self.expert_out,self.optimize],feed_dict={self.expert_inputs:expert_inputs,
                             self.expert_action:expert_actions,
                             self.inputs:inputs,self.action:action,
                             self.predicted_q_value:predicted_q_value,
                             self.expert_q_value:expert_q_value})
    
    def predict_target(self,inputs,action):
        return self.sess.run(self.target_out,feed_dict={self.target_inputs:inputs,self.target_action:action})
    
    def predict(self,inputs,action):
        return self.sess.run(self.out,feed_dict={self.inputs:inputs,self.action:action})
    
    def action_gradients(self,inputs,actions):
        return self.sess.run(self.action_grads,feed_dict={self.inputs:inputs,self.action:actions})
    
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
    
# =============================================================================
#     def update_network_to_expert(self):
#         self.sess.run(self.update_network_to_expert_params)
#     
#     def update_expert_to_network(self):
#         self.sess.run(self.update_expert_to_network_params)
# =============================================================================
    
