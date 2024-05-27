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
        return 'A high-resolution photo of a kitchen.'
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

if __name__ == "__main__":
    # Generate and print a sample prompt
    print(generate_prompt())
