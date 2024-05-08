#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import random
import torch
from pytorch3d.transforms import Transform3d

def read_file(file_path):
    """Read lines from a given file and return them as a list."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

# Load data from files
lead_ins = read_file('./prompts/leaders.txt')
outdoor_scenes = read_file('./prompts/outdoor.txt')
indoor_scenes = read_file('./prompts/indoor.txt')
objects = read_file('./prompts/objects.txt')
arrangements = read_file('./prompts/arrangement.txt')
modifiers = read_file('./prompts/modifiers.txt')
backgrounds = read_file('./prompts/backgrounds.txt')
doors = read_file('./prompts/doors.txt')
facades = read_file('./prompts/facades.txt')

def generate_scene():
    """Generate a random prompt using loaded data."""
    lead_in = random.choice(lead_ins)
    scene = random.choice(indoor_scenes)# + outdoor_scenes) #Combining outdoor and indoor scenes
    #obj = random.choice(objects)
    #arrangement = random.choice(arrangements)
    #background = random.choice(backgrounds)
    #modifier = random.choice(modifiers)

    # Compose the prompt
    prompt = f"{lead_in} a {scene.lower()}"
    #prompt += f" In the foreground, {obj} {arrangement} {background}."
    #prompt += f" This scene is rendered in a {modifier} style."
    return prompt

def generate_doors():
    """The Doors of Perception..."""
    lead_in = random.choice(lead_ins)
    facade = random.choice(facades)# + doors) #Combining portals 
    prompt = f"{lead_in} {facade.lower()}"
    return prompt

def generate_prompt():
    return generate_doors()

def calculate_median_distance(point_cloud, camera_extrinsics):
    """
    Calculates the median distance from the camera to the points in the point cloud using world coordinates.

    Args:
    - point_cloud (torch.Tensor): A tensor of shape (N, 3) containing the coordinates of the points in world coordinates.
    - camera_extrinsics (torch.Tensor): A 4x4 tensor representing the camera extrinsic matrix that transforms points from world coordinates to camera coordinates.

    Returns:
    - float: The median distance of the points from the camera.
    """
    # Extract camera position from the extrinsics matrix
    # The camera position is the negative of the translation components of the matrix, transformed by the rotation part.
    rotation_matrix = camera_extrinsics[:3, :3]
    translation_vector = camera_extrinsics[:3, 3]
    camera_position = -torch.matmul(rotation_matrix.transpose(0, 1), translation_vector)

    distances = torch.norm(point_cloud - camera_position, dim=1)
    return distances.median().item()  # Convert to Python float for general use

def generate_door_control(pts_3d, extrinsics):
    md = calculate_median_distance(pts_3d, extrinsics)
    sequence = []

    trans = md / 3.0
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

def generate_control():
    # choose from moving and new prompts 
    # choose 4-9 1
    rot = random.choice(['a', 'd'])
    trans = random.choice(['s'])

    rot_num = random.choice([2, 3, 4, 5, 6]) 

    if trans == 'w':
        trans_num = random.choice([1, 2])
    else:
        trans_num = random.choice([2, 3, 4])

    sequence = [rot] * rot_num + [trans] * trans_num + ['f']
    return [(x, 0.05) for x in sequence]

if __name__ == "__main__":
    # Generate and print a sample prompt
    print(generate_prompt())
    print(generate_control())
