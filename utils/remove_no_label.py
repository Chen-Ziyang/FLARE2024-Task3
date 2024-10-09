import shutil
import os
import numpy as np
import pickle


### Before training, remove samples without labels
root = '/erwen_SSD/2T/zychen/FLARE2024-Code/nnUNet_preprocessed/Task999_PseudoMR/nnUNetData_plans_FLARE24Small_stage0'
paths = [i for i in os.listdir(root) if 'amos' in i]

for path in paths:
    if 'npy' in path:
        try:
            data = np.load(os.path.join(root, path))[-1]
        except:
            os.remove(os.path.join(root, path))
            continue
        if len(np.unique(data)) <= 2:
            print(path)
            shutil.move(os.path.join(root, path.replace('npy', 'npz')), '/erwen_SSD/2T/zychen/FLARE2024-Code/nnUNet_preprocessed/Task999_PseudoMR/')
            shutil.move(os.path.join(root, path.replace('npy', 'pkl')), '/erwen_SSD/2T/zychen/FLARE2024-Code/nnUNet_preprocessed/Task999_PseudoMR/')
            shutil.move(os.path.join(root, path), '/erwen_SSD/2T/zychen/FLARE2024-Code/nnUNet_preprocessed/Task999_PseudoMR/')

