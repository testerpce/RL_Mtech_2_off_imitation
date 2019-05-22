#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 19:07:41 2019

@author: sayambhu
"""

import tensorflow as tf
import numpy as np
from Actor_critic_imitator import ActorNetwork, CriticNetwork
import gym
from Replay_Buffer_np import ReplayBuffer
from utils_imitate import OrnsteinUhlenbeckActionNoise#,build_summaries

class Imitator:
    
    def __init__(self,env,actor=None,critic=None,actor_lr=0.0001,critic_lr=0.001,
                 gamma=0.99,tau=0.001,actor_bn=True,critic_bn=True,
                 tensorboard_log=None,replay_buffer=None,batch_size=64,
                 replay_buffer_size=50000,expert_buffer=None,
                 expert_dir='TD3_BipedalWalkerv2_Unorm.npz',
                 random_seed=999,summary_dir='./logs',
                 max_episode_len=1000,buffer_size=1000):
        self.env=env
        self.state_dim=env.observation_space.shape[0]
        self.action_dim=env.action_space.shape[0]
        self.actor=actor
        self.critic=critic
        self.actor_lr=actor_lr
        self.critic_lr=critic_lr
        self.gamma=gamma
        self.tau=tau
        self.actor_bn=actor_bn
        self.critic_bn=critic_bn
        self.batch_size=batch_size
        self.buffer_size=buffer_size
        self.random_seed=random_seed
        self.max_episode_len=max_episode_len
        self.tensorboard_log=tensorboard_log
        self.replay_buffer=replay_buffer
        self.expert_buffer=expert_buffer
        self.expert_dir=expert_dir
        self.summary_dir=summary_dir
        
        self.summary_ops,self.summary_vars=self.build_summaries()
        
        self.sess=tf.Session()
        self.setup_models()
    
    def setup_models(self):
        if self.actor is None:
            print("setting up actor")
            self.actor=ActorNetwork(sess=self.sess,
                               state_dim=self.env.observation_space.shape[0],
                               action_dim=self.env.action_space.shape[0],
                               action_bound=self.env.action_space.high,
                               learning_rate=self.actor_lr,
                               tau=self.tau,
                               batch_size=self.batch_size,
                               actor_bn=self.actor_bn)
        if self.critic is None:
            print("setting up critic")
            self.critic=CriticNetwork(sess=self.sess,
                                      state_dim=self.env.observation_space.shape[0],
                                      action_dim=self.env.action_space.shape[0],
                                      learning_rate=self.critic_lr,
                                      tau=self.tau,
                                      gamma=self.gamma,
                                      num_actor_vars=self.actor.get_num_trainable_vars(),
                                      critic_bn=self.critic_bn)
        
        if self.replay_buffer is None:
            print("Setting up replay buffer")
            self.replay_buffer=ReplayBuffer(state_dim=self.state_dim,
                                            action_dim=self.action_dim,
                                            buffer_size=self.buffer_size,
                                            random_seed=self.random_seed)
        
        if self.expert_buffer is None:
            assert(self.expert_dir is not None)
            print("setting up expert buffer")
            data=np.load(self.expert_dir)
            expert_len=len(data['obs'])
            self.expert_buffer=ReplayBuffer(state_dim=self.state_dim,
                                            action_dim=self.action_dim,
                                            buffer_size=expert_len,
                                            random_seed=self.random_seed)
            
            #Ignore the terminal part it says episode starts when replay 
            #buffer is actually storing whether it is terminal or not
# =============================================================================
#             for i in range(len(data['obs'])-1):
#                 s=data['obs'][i]
#                 a=data['actions'][i]
#                 r=data['rewards'][i]
#                 t= 0#data['episode_starts'][i+1]
#                 s2=data['obs'][i+1]
#                 self.expert_buffer.add(s,a,r,t,s2)
# =============================================================================
            
            self.expert_buffer.expert_add(s=data['obs'],a=data['actions'],r=data['rewards'],t=data['episode_dones'],s2=data['obs_next'])
            
                
    
    def build_summaries(self):
        print("Building summaries")
        episode_reward=tf.Variable(0.)
        tf.summary.scalar('episode_reward',episode_reward)
        episode_ave_max_q=tf.Variable(0.)
        tf.summary.scalar('Qmax_value',episode_ave_max_q)
        episode_expert_ave_max_q=tf.Variable(0.)
        tf.summary.scalar('Expert_Qmax_value',episode_expert_ave_max_q)
        summary_vars=[episode_reward,episode_ave_max_q,episode_expert_ave_max_q]
        summary_ops=tf.summary.merge_all()
        
        return summary_ops,summary_vars
        
    
    def train(self,num_episodes,actor_noise,writer):
        #build summaries, initialize global variables, make the writer
        #and first update actor and critic
        #Also initiate the expert buffer
        # for episodes: reset environment and set reward and ave max_q to
        #be zero now for max episode length: predict for current state the action
        #and store the result of that in buffer
        #If the buffer size is greater than the minibatch size, then
        #Sample from the buffer and the expert buffer
        #Train the critic. Use the gradients from the current states batch
        #Use it to get action gradients and train the actor
        # If terminal check the reward and break
        print ("Getting into training")
        for i in range(num_episodes):
            s=self.env.reset()
            ep_reward=0
            ep_ave_max_q=0
            ep_ave_max_q_expert=0
            
            for j in range(self.max_episode_len):
                
                a=self.actor.predict(np.reshape(s,(1,self.actor.s_dim)))+actor_noise()
# =============================================================================
#                 print(a[0])
# =============================================================================
                s2,r,terminal,info=self.env.step(a[0])
                
                self.replay_buffer.add(np.reshape(s,(self.actor.s_dim,)),np.reshape(a,(self.actor.a_dim,)),r,terminal,np.reshape(s2,(self.actor.s_dim,)))
                
                if self.replay_buffer.size()>self.batch_size :
                    
                    s_batch,a_batch,r_batch,s2_batch=self.replay_buffer.sample_batch(self.batch_size)
                    expert_s_batch,expert_a_batch,expert_r_batch,expert_s2_batch=self.expert_buffer.sample_batch(self.batch_size)
                    
                    target_q=self.critic.predict_target(s2_batch,self.actor.predict_target(s2_batch))
                    
                    expert_target_q=self.critic.predict_target(expert_s2_batch,self.actor.predict_target(expert_s2_batch))
                    
                    predicted_q_value,expert_q_value,_=self.critic.train(expert_s_batch,expert_a_batch,s_batch,a_batch,target_q,expert_target_q)
# =============================================================================
#                     print("predicted_q_value = ",predicted_q_value,"exper_q_value",expert_q_value)
# =============================================================================
                    
                    ep_ave_max_q+=np.amax(predicted_q_value)
                    ep_ave_max_q_expert+=np.amax(expert_q_value)
                    
                    a_outs=self.actor.predict(s_batch)
                    grads=self.critic.action_gradients(s_batch,a_outs)
                    
                    self.actor.train(s_batch,grads[0])
                    
                    self.actor.update_target_network()
                    self.critic.update_target_network()
                    
                s=s2
                ep_reward+=r
                
                if terminal:
# =============================================================================
#                     print("ep_reward = ",ep_reward,"ep_ave_max_q/float(j) = ",ep_ave_max_q/float(j),"ep_ave_max_q_expert/float(j) = ",ep_ave_max_q_expert/float(j))
# =============================================================================
                    summary_str=self.sess.run(self.summary_ops,feed_dict={self.summary_vars[0]:ep_reward,self.summary_vars[1]:ep_ave_max_q/float(j),self.summary_vars[2]:ep_ave_max_q_expert/float(j)})
                    writer.add_summary(summary_str,i)
                    writer.flush()
                    print('| Reward:{:d} | Episode:{:d}| Qmax:{:.4f} | Expert_Qmax:{:.4f}'.format(int(ep_reward),i,(ep_ave_max_q/float(j)),(ep_ave_max_q_expert/float(j))))
                    break
                        
                    
                    #This is where you stopped
            
            
    
    def learn(self,num_episodes=1000):
        
        print("Getting into learn")
        #call the train function I guess
        self.sess.run(tf.global_variables_initializer())
        
        writer=tf.summary.FileWriter(self.summary_dir,self.sess.graph)
        self.actor.update_target_network()
        self.critic.update_target_network()
        
        self.actor_noise=OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.env.action_space.shape[0]))
        
        
        self.train(num_episodes=num_episodes,actor_noise=self.actor_noise,writer=writer)
        
        
        
        





# =============================================================================
# env=gym.make('BipedalWalker-v2')
# state_dim=env.observation_space.shape[0]
# action_bound=env.action_space.high
# =============================================================================

        
        
    
    
        
        
        
        
        
