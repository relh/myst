#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

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

def generate_prompt():
    """Generate a random prompt using loaded data."""
    scene = random.choice(outdoor_scenes + indoor_scenes)  # Combining outdoor and indoor scenes
    obj = random.choice(objects)
    arrangement = random.choice(arrangements)
    background = random.choice(backgrounds)
    lead_in = random.choice(lead_ins)
    #modifier = random.choice(modifiers)

    # Compose the prompt
    prompt = f"{lead_in} a {scene.lower()}"
    #prompt += f" In the foreground, {obj} {arrangement} {background}."
    #prompt += f" This scene is rendered in a {modifier} style."
    return prompt

if __name__ == "__main__":
    # Generate and print a sample prompt
    print(generate_prompt())

