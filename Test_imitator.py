#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:29:46 2019

@author: sayambhu
"""

from Imitator import Imitator
import gym

env=gym.make('BipedalWalker-v2')

imit=Imitator(env,max_episode_len=10000)

imit.learn()
