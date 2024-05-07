#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import torch
from pytorch3d.transforms import RotateAxisAngle, Translate

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

def calculate_median_distance(point_cloud):
    # Calculate the median distance of points in the point cloud from the origin
    distances = torch.norm(point_cloud, dim=1)
    return distances.median()

def move_camera(initial_extrinsics, distance):
    # Translate back to 2x the median distance
    move_back = Translate((0, 0, distance))  # Moving additional 1x distance back
    back_extrinsics = move_back.transform_matrix @ initial_extrinsics

    # Translate left and rotate right
    move_left = Translate((-distance, 0, 0))  # Moving 1x distance to the left
    rotate_right = RotateAxisAngle(angle=45, axis="Y")
    left_rotate_extrinsics = rotate_right.transform_matrix @ move_left.transform_matrix @ back_extrinsics

    # Move back to the center and then right and rotate left
    move_right = Translate((distance, 0, 0))  # Moving 1x distance to the right
    rotate_left = RotateAxisAngle(angle=-45, axis="Y")
    right_rotate_extrinsics = rotate_left.transform_matrix @ move_right.transform_matrix @ back_extrinsics

    return back_extrinsics, left_rotate_extrinsics, right_rotate_extrinsics

def generate_door_control(pts_3d, extrinsics):
    median_distance = calculate_median_distance(point_cloud)
    extrinsics_b, extrinsics_lrr, extrinsics_rrl = move_camera(extrinsics, median_distance)
    return [extrinsics_b, extrinsics_lrr, extrinsics_rrl]

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
    return sequence

if __name__ == "__main__":
    # Generate and print a sample prompt
    print(generate_prompt())
    print(generate_control())
