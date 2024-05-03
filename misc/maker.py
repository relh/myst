#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

def read_file(file_path):
    """Read lines from a given file and return them as a list."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

# Load data from files
lead_ins = read_file('./assets/prompts/leaders.txt')
outdoor_scenes = read_file('./assets/prompts/outdoor.txt')
indoor_scenes = read_file('./assets/prompts/indoor.txt')
objects = read_file('./assets/prompts/objects.txt')
arrangements = read_file('./assets/prompts/arrangement.txt')
modifiers = read_file('./assets/prompts/modifiers.txt')
backgrounds = read_file('./assets/prompts/backgrounds.txt')

def generate_control():
    # choose from moving and new prompts 
    # choose 4-9 1
    rot = random.choice(['a', 'd'])
    trans = random.choice(['w', 's'])

    rot_num = random.choice([2, 3, 4, 5, 6]) 

    if trans == 'w':
        trans_num = random.choice([1, 2])
    else:
        trans_num = random.choice([2, 3, 4])

    sequence = [rot] * rot_num + [trans] * trans_num + ['f']
    return sequence

def generate_prompt():
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

if __name__ == "__main__":
    # Generate and print a sample prompt
    print(generate_prompt())
    print(generate_control())
