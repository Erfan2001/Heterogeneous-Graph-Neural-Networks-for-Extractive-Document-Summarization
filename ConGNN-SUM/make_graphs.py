import os
import time
offset = 1280
for i in range(12800,13000,offset):
    os.system(f"python evaluation.py --from_instances_index {i} --max_instances {offset}")
    # os.system(f"python train.py --from_instances_index {i} --max_instances {offset}")
