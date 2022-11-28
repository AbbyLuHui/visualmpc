#!/usr/bin/env python3

"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""
import os
import sys
import argparse
import pyglet
import math
from pyglet.window import key
from pyglet import clock
import numpy as np
import gym
import gym_miniworld
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-Square-v0')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--no-time-limit', action='store_true', help='ignore time step limits')
parser.add_argument('--top_view', action='store_true', help='show the top view instead of the agent view')
parser.add_argument('--dataset_size', type=int, default=100)
args = parser.parse_args()


index = 0
retry = 0
env = gym.make(args.env_name)
root_dir = './data/'
while index < args.dataset_size:
    try:
        if retry > 20:
            break
        # kwargs={"exp_index": index}


        if args.no_time_limit:
            env.max_episode_steps = math.inf
        if args.domain_rand:
            env.domain_rand = True

        view_mode = 'top' if args.top_view else 'agent'

        actions = []
        while len(actions) <= 1:
            _, actions, dir, no_detour, path_count = env.reset(exp_index=index, root_dir=root_dir)
            #if index % 2 == 0 and no_detour:
            #    actions = []

            #if index % 2 == 1 and not no_detour:
            #    actions = []

        # Create the display window
        env.render('rgb_array', view=view_mode)

        prev_action = None

        for act in actions:
            if prev_action == None: # first step
                if act == 2 and dir == 0:
                    env.step(env.actions.turn_right)
                elif act == 4 and dir == -math.pi / 2:
                    env.step(env.actions.turn_left)

            elif act != prev_action:
                if (prev_action == 2 and act == 4) or (prev_action == 4 and act == 1) or \
                    (prev_action == 1 and act == 3) or (prev_action == 3 and act == 2):
                    env.step(env.actions.turn_left)
                if (prev_action == 4 and act == 2) or (prev_action == 2 and act == 3) or \
                    (prev_action == 3 and act == 1) or (prev_action == 1 and act == 4):
                    env.step(env.actions.turn_right)
            if act != 0:
                env.step(env.actions.move_forward)
                prev_action = act
        np.save('./data/actions/action' + str(index).zfill(5) + '.npy', np.array(actions))


        if (index % 1000) == 0:
            print("Generated {} data samples".format(index))
        index += 1
        retry = 0
    except Exception as e:
        print(e)
        retry += 1

env.close()
if os.path.exists('./data/maps/map00000.npy'):
    os.remove('./data/maps/map00000.npy')
if os.path.exists('./data/maps_gt_obs/map00000.npy'):
    os.remove('./data/maps_gt_obs/map00000.npy')
if os.path.exists('./data/actions/action00000.npy'):
    os.remove('./data/actions/action00000.npy')
