#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import random

import torch
from pytorch3d.transforms import Transform3d

from misc.scale import median_scene_distance 


def read_file(file_path):
    """Read lines from a given file and return them as a list."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip()+' ' for line in file if line.strip()]

# arrangement 
cameras = read_file('./prompts/camera.txt')
vantages = read_file('./prompts/vantages.txt')
lead_ins = read_file('./prompts/leaders.txt')
arrangements = read_file('./prompts/arrangement.txt')
objects = read_file('./prompts/objects.txt')

# scenes
outdoors = read_file('./prompts/outdoor.txt')
indoors = read_file('./prompts/indoor.txt')
backgrounds = read_file('./prompts/backgrounds.txt')

# doors
doors = read_file('./prompts/doors.txt')
facades = read_file('./prompts/facades.txt')

def generate_prompt(prompt):
    """Generate a random prompt using loaded data."""

    if prompt == 'me':
        return input(f"enter stable diffusion initial scene: ")
    elif prompt == 'default':
        return 'A high-resolution image of a kitchen.'
    elif prompt == 'doors':
        lead_in = random.choice(lead_ins)
        facade = random.choice(facades)# + doors) #Combining portals 
        return f"{lead_in} {facade.lower()}"
    elif prompt == 'auto':
        lead_in = random.choice(lead_ins)
        scene = random.choice(indoors)# + outdoor_scenes) #Combining outdoor and indoor scenes
        return f"{lead_in} a {scene.lower()}"
    elif prompt == 'combo':
        camera = random.choice(cameras + ['']*20)
        vantage = random.choice(vantages + ['']*20)
        arrangement = random.choice(arrangements)
        foreground = random.choice(objects)
        background = random.choice(backgrounds + outdoors + indoors)
        return f"A {camera}photograph {vantage}at {foreground[:-1]} {arrangement[:-1]} {background.lower()[:-1]}"

def generate_control(control, amount):
    # choose from moving and new prompts 
    # choose 4-9 1

    if control == 'auto':
        rot = random.choice(['a', 'd'])
        trans = random.choice(['w', 's', 's', 's'])

        rot_num = random.choice([2, 3, 4, 5]) 

        if trans == 'w':
            trans_num = random.choice([1, 2, 3])
        else:
            trans_num = random.choice([2, 3, 4, 5, 6, 7])

        sequence = [rot] * rot_num + [trans] * trans_num + ['f']
        return [(x, amount) for x in sequence]

    elif control == 'doors':
        sequence = []
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
        return sequence

if __name__ == "__main__":
    # Generate and print a sample prompt
    print(generate_prompt())
    print(generate_control())
