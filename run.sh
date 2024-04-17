#!/bin/bash
srun --account=eel6763 --qos=eel6763 -p gpu --gpus=1 --time=03:00:00  --pty -u bash -i
module load ufrc
module load class/eel6763
module load cuda