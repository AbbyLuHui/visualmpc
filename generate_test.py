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

import torch
import torch.nn as nn
from collections import OrderedDict
from PIL import Image
import torchvision.transforms as transforms
import crypten
from multiprocess_launcher import MultiProcessLauncher
#import crypten.mpc as mpc
import crypten.communicator as comm

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-Square-v0')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--no-time-limit', action='store_true', help='ignore time step limits')
parser.add_argument('--top_view', action='store_true', help='show the top view instead of the agent view')
parser.add_argument('--root_dir', default='./logs/')
args = parser.parse_args()
MAX_STEP = 17

class plaintextNet(nn.Module):
    def __init__(self, multi_view=5):
        super().__init__()
        self.multi_view = multi_view

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 6, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(6, 6, 5, stride=2),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(648, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.map_encoding = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.action_predict = nn.Sequential(
            nn.Linear(32 * multi_view + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        )

    def forward(self, x, maps, fov):
        B, V, C, H, W = x.shape
        observations = []
        if self.multi_view > 1:
            for i in range(V):
                obs = self.cnn(x[:, i])
                observations.append(obs)
        if self.multi_view > 0:
            observations.append(self.cnn(fov))
        map_encode = self.map_encoding(maps)
        observations.append(map_encode)
        embed = torch.cat(observations, dim=1)
        return self.action_predict(embed)

class featureNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 6, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(6, 6, 5, stride=2),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(648, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

    def forward(self, x):
        return self.cnn(x)


class mapNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.map_encoding = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, maps):
        return self.map_encoding(maps)

class cryptenNet(crypten.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_predict = crypten.nn.Sequential(
            crypten.nn.Linear(288, 128),
            crypten.nn.ReLU(),
            crypten.nn.Linear(128, 64),
            crypten.nn.ReLU(),
            crypten.nn.Linear(64, 16),
            crypten.nn.ReLU(),
            crypten.nn.Linear(16, 5)
        )

    def forward(self, x):
        return self.action_predict(x)


def get_items(index, start_view='random', root_dir=None):
    images = []
    for subfolder in ['bottom_left', 'bottom_right', 'top_left', 'top_right', 'agent_view']:
        img_loc = root_dir + '/' + subfolder + '/img' + str(index).zfill(5) + '.png'
        image = Image.open(img_loc).convert("RGB")
        tensor_image = transforms.ToTensor()(image)
        images.append(tensor_image)

    map_loc = root_dir + '/maps/map' + str(index).zfill(5) + '.npy'
    map = torch.from_numpy(np.load(map_loc, allow_pickle=True))
    map_gt_loc = root_dir + '/maps_gt_obs/map' + str(index).zfill(5) + '.npy'
    map_gt = torch.from_numpy(np.load(map_gt_loc, allow_pickle=True))
    start_view_index = 2 if start_view=='deterministic' else 4
    return torch.stack(images[:4], 0).unsqueeze(0), map.float().unsqueeze(0), \
           map_gt.float().unsqueeze(0), images[start_view_index].unsqueeze(0)

def get_encoders(pretrain_dir):
    cnn = featureNet()
    map_encoder = mapNet()

    pretrained_dict = torch.load(pretrain_dir, map_location=torch.device('cpu'))
    cnn_dict = cnn.state_dict()
    map_encoder_dict = map_encoder.state_dict()
    cnn_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in cnn_dict}

    if len(cnn_pretrained_dict) == 0:
        pretrained_dict_rename = OrderedDict()
        for k, v in pretrained_dict.items():
            name = k[7:]
            pretrained_dict_rename[name] = v
        pretrained_dict = pretrained_dict_rename

    cnn_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in cnn_dict}
    map_encoder_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in map_encoder_dict}

    cnn_dict.update(cnn_pretrained_dict)
    map_encoder_dict.update(map_encoder_pretrained_dict)

    cnn.load_state_dict(cnn_pretrained_dict)
    map_encoder.load_state_dict(map_encoder_pretrained_dict)

    cnn.eval()
    map_encoder.eval()
    return cnn, map_encoder

def update_maps(predicted, maps, maps_gt):
    # assume square map
    map_length = maps.shape[-1]
    # currently taking in predicted value as ground truth
    current_pos = (maps == 1).nonzero(as_tuple=False)
    # 1 - left, 2 - right, 3 - up, 4 - down

    action_updates = torch.tensor([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)])
    actions = torch.index_select(action_updates, 0, predicted)

    new_maps = maps.clone()
    new_maps[current_pos[:, 0], current_pos[:, 1], current_pos[:, 2]] = 0
    next_pos = current_pos.clone()
    next_pos[:, 1] += actions[:, 0]
    next_pos[:, 2] += actions[:, 1]

    next_pos_masked = torch.where(torch.logical_or(next_pos < 0, next_pos >= map_length), current_pos, next_pos)
    next_pos_masked = torch.where(torch.logical_or(maps_gt[next_pos_masked[:, 0], next_pos_masked[:, 1], next_pos_masked[:, 2]]==3,
                                                     maps_gt[next_pos_masked[:, 0], next_pos_masked[:, 1], next_pos_masked[:, 2]]==4),
                                  current_pos, next_pos_masked).clone()
    valid = (next_pos_masked == next_pos).all()
    new_maps[next_pos_masked[:, 0], next_pos_masked[:, 1], next_pos_masked[:, 2]] = 1
    return new_maps, valid

#@mpc.run_multiprocess(world_size=2)
def test(pretrained_crypt, pretrained_plain, data_size, model_type, multi_view=5, mode='reach_goal', random=False, start_view='deterministic', root_dir='./logs/', no_party=2):
    #if not os.path.isdir(root_idr):
    #    os.mkdir(root_dir)
    assert mode in ['reach_goal', 'no_obs_on_path', 'obs_on_path', 'efficiency']

    net, cnn, map_encoder = None, None, None
    if model_type == 'plaintext':
        net = plaintextNet(multi_view=multi_view)
        pretrained_dict = torch.load(pretrained_plain, map_location=torch.device('cpu'))
        pretrained_dict_rename = OrderedDict()
        for k, v in pretrained_dict.items():
            name = k[7:]
            pretrained_dict_rename[name] = v
        pretrained_dict = pretrained_dict_rename
        net.load_state_dict(pretrained_dict)
    elif model_type == 'crypten':
        net = torch.load(pretrained_crypt, map_location=torch.device('cpu')).encrypt()
        cnn, map_encoder = get_encoders(pretrained_plain)

    net.eval()

    index = 0
    env = gym.make(args.env_name)

    success = 0
    total = 0
    efficient = 0
    error_index = []
    rank = 0 if model_type == 'plaintext' else comm.get().get_rank()

    while total < data_size:
        if os.path.isdir(os.path.join(root_dir, 'actions/')):
            index = len(os.listdir(os.path.join(root_dir, 'actions/')))

        if args.no_time_limit:
            env.max_episode_steps = math.inf
        if args.domain_rand:
            env.domain_rand = True
        env.img_index = 0

        view_mode = 'top' if args.top_view else 'agent'

        actions = []
        while len(actions) <= 1 and rank == 0:
            _, actions, dir, no_detour, path_count = env.reset(exp_index=index, root_dir=root_dir)
            if mode == 'reach_goal':
                if index % 2 == 0 and no_detour:
                    actions = []
                if index % 2 == 1 and not no_detour:
                    actions = []
            elif mode == 'no_obs_on_path':
                if not no_detour:
                    actions = []
            elif mode == 'obs_on_path':
                if no_detour:
                    actions = []
            elif mode == 'efficiency':
                if path_count == 1:
                    actions = []
                if index % 2 == 0 and no_detour:
                    actions = []
                if index % 2 == 1 and not no_detour:
                    actions = []
            else:
                print("INVALID MODE")

        # Create the display window
        if rank == 0:
            env.render('rgb_array', view=view_mode)

        crypten.save_from_party(torch.tensor(actions), root_dir + '/actions/action' + str(index).zfill(5) + '.npy', src=0)

        if index != 0:
            total += 1
            prev_action, reached = None, 0
            predicted_path = []
            images, maps, maps_gt, fov = get_items(index, start_view=start_view, root_dir=root_dir)
            if model_type == 'crypten':
                for party in range(images.shape[1]):
                    feat = cnn(images[:, party])
                    src = 1 if no_party == 2 else party + 1
                    crypten.save_from_party(feat, root_dir + '/features/feat' + str(party) + str(index).zfill(5) + '.pth', src=src)

            act = -1
            step = 0
            while step < MAX_STEP:
                if act == 0:
                    step += 1
                    if reached and step == MAX_STEP - 1:
                        pass
                    continue


                if model_type == 'plaintext':
                    output = net(images, maps, fov)
                elif model_type == 'crypten':
                    crypten.save_from_party(cnn(fov), root_dir + '/features/feat_fov_' + str(index).zfill(5) + '.pth', src=0)
                    crypten.save_from_party(map_encoder(maps), root_dir + '/features/feat_map_' + str(index).zfill(5) + '.pth', src=0)
                    img_feat = []
                    for party in range(images.shape[1]):
                        src = 1 if no_party == 2 else party + 1
                        feat = crypten.load_from_party(root_dir + '/features/feat' + str(party) + str(index).zfill(5) + '.pth', src=src)
                        img_feat.append(feat)
                    feat_fov = crypten.load_from_party(root_dir + '/features/feat_fov_' + str(index).zfill(5) +'.pth', src=0)
                    img_feat.append(feat_fov)
                    feat_map = crypten.load_from_party(root_dir + '/features/feat_map_' + str(index).zfill(5) + '.pth', src=0)
                    img_feat.append(feat_map)
                    input = crypten.cat(img_feat, dim=1)
                    output = net(input).get_plain_text()

                _, predicted = torch.max(output, dim=1)

                if random:
                    predicted = torch.randint(1, 5, (1,))
                    maps, valid = update_maps(predicted, maps, maps_gt)
                    if not valid:
                        continue
                else:
                    maps, valid = update_maps(predicted, maps, maps_gt)


                # update fov
                act = predicted.item()
                #crypten.print(act)
                if rank == 0:
                    predicted_path.append(act)

                if prev_action == None and rank == 0:  # first step
                    if act == 2 and dir == 0:
                        env.step(env.actions.turn_right)
                    elif act == 4 and dir == -math.pi / 2:
                        env.step(env.actions.turn_left)

                elif act != prev_action and rank == 0:
                    if (prev_action == 2 and act == 4) or (prev_action == 4 and act == 1) or \
                            (prev_action == 1 and act == 3) or (prev_action == 3 and act == 2):
                        env.step(env.actions.turn_left)
                    if (prev_action == 4 and act == 2) or (prev_action == 2 and act == 3) or \
                            (prev_action == 3 and act == 1) or (prev_action == 1 and act == 4):
                        env.step(env.actions.turn_right)
                    if (prev_action == 1 and act == 2) or (prev_action == 2 and act == 1) or \
                            (prev_action == 3 and act == 4) or (prev_action == 4 and act == 3):
                        env.step(env.actions.turn_right)
                        env.step(env.actions.turn_right)

                if act != 0 and rank == 0:
                    fov, _, _, _ = env.step(env.actions.move_forward)
                    fov = transforms.ToTensor()(Image.fromarray(fov).convert('RGB')).unsqueeze(0)
                    prev_action = act
                if not (maps == 2).any() and act == 0:
                    reached = 1
                    success += 1
                    if len(actions) >= len(predicted_path):
                        efficient += 1
                elif not (maps == 2).any() and random == True:
                    reached = 1
                    success += 1
                    act = 0
                    if len(actions) >= len(predicted_path):
                        efficient += 1
                elif (act==0 and (maps == 2).any()) or (step == MAX_STEP - 1 and act != 0):
                    if rank == 0:
                        error_index.append(index)

                if act == 0 or step == MAX_STEP-1:
                    crypten.save_from_party(torch.tensor(predicted_path), root_dir + '/predicted/action' + str(index).zfill(5) + '.npy', src=0)

                step += 1
        if (index % 1000) == 0:
            print("Generated {} data samples".format(index))
        index += 1

    env.close()
    crypten.print('-------------')
    crypten.print('model type: ', model_type)
    crypten.print('test mode: ', mode)
    crypten.print('multiview: ', multi_view)
    crypten.print('random: ', random)
    crypten.print('success rate: ', success / total)
    crypten.print('efficient rate: ', efficient / max(success, 1))
    crypten.print('error index: ', error_index)
    with open(root_dir + '/log.txt', 'w') as f:
        f.write('test on dataset mode: ' + mode +'\n')
        f.write('random action: ' + str(random) + '\n')
        f.write('model type: ' + model_type + '\n')
        if model_type == 'crypten':
            f.write('pretrained crypt: ' + pretrained_crypt + '\n')
        f.write('pretrained plain: ' + pretrained_plain + '\n')
        f.write('success cases: ' + str(success) + '\n')
        f.write('total cases: ' + str(total) + '\n')
        f.write('success rate: ' + str(success / total) + '\n')
        #if mode == 'efficiency':
        f.write('efficiency rate: ' + str(efficient / max(success, 1)) + '\n')
        f.write('error index: ' + str(error_index) + '\n')
    f.close()

def start(party, model_type='crypten', multi_view=5, data_size=2250, random=False, start_view='random', exp_dir='./logs/ciphertext_two_party'):
    if party == 2:
        pretrained_crypt = './models/two_party.pth'
    elif party == 5:
        pretrained_crypt = './models/five_party.pth'
        exp_dir = './logs/ciphertext_five_party'
    if multi_view == 0:
        pretrained_plain = './models/map_only.pth'
    elif multi_view == 1 and start_view == 'random':
        pretrained_plain = './models/single_view_random.pth'
    elif multi_view == 1 and start_view == 'deterministic':
        pretrained_plain = './models/single_view_deterministic.pth'
    elif multi_view == 5:
        pretrained_plain = './models/multi_view.pth'

    rank = 0 if model_type == 'plaintext' else comm.get().get_rank()
    for mode in ['reach_goal', 'efficiency', 'no_obs_on_path', 'obs_on_path']:
        sub_exp_dir = os.path.join(exp_dir, mode)
        if rank == 0 and not os.path.isdir(sub_exp_dir):
            os.mkdir(sub_exp_dir)
        if rank == 0 and not os.path.isdir(os.path.join(sub_exp_dir, 'features/')):
            os.mkdir(os.path.join(sub_exp_dir, 'features/'))
        if rank == 0 and not os.path.isdir(os.path.join(sub_exp_dir, 'predicted/')):
            os.mkdir(os.path.join(sub_exp_dir, 'predicted/'))

        test(pretrained_crypt, pretrained_plain, data_size, model_type, multi_view=multi_view, mode=mode, random=random, start_view=start_view, root_dir=sub_exp_dir, no_party=party)

        if rank == 0 and os.path.exists(os.path.join(sub_exp_dir, 'maps/map00000.npy')):
            os.remove(os.path.join(sub_exp_dir, 'maps/map00000.npy'))
        if rank == 0 and os.path.exists(os.path.join(sub_exp_dir, 'maps_gt_obs/map00000.npy')):
            os.remove(os.path.join(sub_exp_dir, 'maps_gt_obs/map00000.npy'))
        if rank == 0 and os.path.exists(os.path.join(sub_exp_dir, 'actions/action00000.npy')):
            os.remove(os.path.join(sub_exp_dir, 'actions/action00000.npy'))

if __name__ == '__main__':
    crypten.init()
    root_dir = args.root_dir

    # plaintext single view deterministic start view
    exp_dir = os.path.join(root_dir, 'plaintext_single_view_deterministic_start')
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    start(2, model_type='plaintext', multi_view=1, start_view='deterministic', exp_dir=exp_dir)
    """
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)

    view_no_to_file_name = {0: 'plaintext_map_only',
                            1: 'plaintext_single_view',
                            5: 'plaintext_multi_view'}

    # plaintext test
    for multi_view in [0, 1, 5]:
        exp_dir = os.path.join(root_dir, view_no_to_file_name[multi_view])
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)
        start(2, model_type='plaintext', multi_view=multi_view, exp_dir=exp_dir)

    # random baseline
    exp_dir = os.path.join(root_dir, 'random')
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    start(2, model_type='plaintext', exp_dir=exp_dir, random=True)

    # crypten training
    exp_dir = os.path.join(root_dir, 'ciphertext_two_party')
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    launcher = MultiProcessLauncher(2, start, 2)
    launcher.start()
    launcher.join()
    launcher.terminate()

    exp_dir = os.path.join(root_dir, 'ciphertext_five_party')
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    launcher = MultiProcessLauncher(5, start, 5)
    launcher.start()
    launcher.join()
    launcher.terminate()"""

