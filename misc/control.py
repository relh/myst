#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import os
import pickle
import random
import sys
import termios
import tty
from argparse import ArgumentParser

import torch
from pytorch3d.transforms import Transform3d

from misc.scale import median_scene_distance

def get_keypress():
    fd = sys.stdin.fileno()
    original_attributes = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)  # Read a single character
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_attributes)
    return key

def generate_control(control, amount, idx):
    if idx == 0:
        # choose from moving and new prompts 
        # choose 4-9 1
        sequence = []
        if control == 'auto':
            rot = random.choice(['a', 'd'])
            trans = random.choice(['w', 's', 's', 's'])

            rot_num = random.choice([2, 3, 4, 5]) 

            if trans == 'w':
                trans_num = random.choice([1, 2, 3])
            else:
                trans_num = random.choice([2, 3, 4, 5, 6, 7])

            sequence = [rot] * rot_num + [trans] * trans_num + ['f']
            sequence = [(x, amount) for x in sequence]

        elif control == 'doors':
            trans = amount / 3.0
            rot = math.pi / 3.0

            for steps in [1, 2]:  
                # 1. first move left or right 1x scene distance
                motion = random.choice(['a', 'd'])
                sequence.append((motion, math.pi / 2.0))
                sequence.append(('s', trans))

                # 2. orient to the scene again and fill
                sequence.append((motion, -rot))
                sequence.append(('f', None))

                # 3. go to the other side and fill
                sequence.append((motion, rot))
                sequence.append(('w', trans))
                sequence.append(('w', trans))
                sequence.append((motion, -rot * 2.0))
                sequence.append(('f', None))

                sequence.append((motion, -rot * 1.0))
                sequence.append(('w', trans))

                sequence.append((motion, math.pi / 2.0))
                sequence.append(('s', trans))
                sequence.append(('f', None))

            # 4. go to the middle and reverse again
            #sequence.append('f', median_distance)
        print(f'ai sequence is... {sequence}')

    if control == 'me':
        return get_keypress(), None
    else:
        if idx >= len(sequence): return 'k', None
        user_input, scale = sequence[idx]
    return user_input, scale

if __name__ == "__main__":
    # Generate and print a sample sequence 
    print(generate_control())
