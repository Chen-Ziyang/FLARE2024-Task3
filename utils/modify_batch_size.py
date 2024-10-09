import shutil
import os
import numpy as np
import pickle


### Set batch_size=4 for MR-Net
root = '/erwen_SSD/2T/zychen/FLARE2024-Code/nnUNet_preprocessed/Task999_PseudoMR/nnUNetPlansFLARE24Big_plans_3D.pkl'
with open(root, 'rb') as file:
    data = pickle.load(file)
data['plans_per_stage'][0]['batch_size'] = 4
with open(root, 'wb') as file:
    pickle.dump(data, file)

### Set batch_size=16 for LW-Net
root = '/erwen_SSD/2T/zychen/FLARE2024-Code/nnUNet_preprocessed/Task999_PseudoMR/nnUNetPlansFLARE24Small_plans_3D.pkl'
with open(root, 'rb') as file:
    data = pickle.load(file)
data['plans_per_stage'][0]['batch_size'] = 16
with open(root, 'wb') as file:
    pickle.dump(data, file)