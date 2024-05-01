#!/bin/bash

for i in {1..35}
do
  python run.py
  pkill python
done
