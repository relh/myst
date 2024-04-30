#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random


def generate_control():
    # choose from moving and new prompts
    # choose 4-9 1
    rot = random.choice(['a', 'd'])
    trans = random.choice(['w', 's'])

    rot_num = random.choice([4, 5, 6, 7, 8, 9]) 
    trans_num = random.choice([3, 4, 5, 6])

    sequence = [rot] * rot_num + [trans] * trans_num
    return sequence

if __name__ == "__main__": 
    print(generate_control())

