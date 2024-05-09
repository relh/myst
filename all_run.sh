#!/bin/bash

for i in {1..35}
do
  python misc/write.py
  python run.py
  pkill python
done
