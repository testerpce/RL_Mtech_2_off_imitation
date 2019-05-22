#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:29:46 2019

@author: sayambhu
"""

from Imitator import Imitator
# =============================================================================
# import os
# from utils_imitate import get_saved_hyperparams,create_test_env
# =============================================================================
print ("\n Imitator imported \n")

################################
#Time to normalize the environment
import gym
env_id='BipedalWalker-v2'
# =============================================================================
# stats_path=os.path.join(os.getcwd(),env_id)
# hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)
# algo_id='Imitator'
# print("hyperparams here= ",hyperparams)
# 
# log_dir=os.path.join(os.getcwd(),'logs',algo_id,env_id)
# env = create_test_env(env_id,n_envs=1, is_atari=False,log_dir=log_dir,
#                           stats_path=stats_path, seed=1000,
#                           should_render=True,
#                           hyperparams=hyperparams)
# =============================================================================
env=gym.make('BipedalWalker-v2')

###############################
print("\n Gym made \n")
imit=Imitator(env,max_episode_len=2000)
print("\n Imitator made \n")
imit.learn()
print("\n imitator learn done \n")
