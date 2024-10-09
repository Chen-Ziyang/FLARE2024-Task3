import pickle
from collections import OrderedDict


### For Fine-tuned Uni-Net
# Before prediction, modify DDP to normal
root = '/data1/userdisk0/zychen/RESULTS_FOLDER/nnUNet/3d_fullres/' \
       'Task666_FLARE2024Challenge/nnUNetTrainerV2_FLARE_Pseudo__nnUNetPlansFLARE24Big/all/model_final_checkpoint.model.pkl'
with open(root, 'rb') as file:
    data = pickle.load(file)
data['init'] = ('/data1/userdisk0/zychen/RESULTS_FOLDER/nnUNet/3d_fullres/Task666_FLARE2024Challenge/'
                'nnUNetTrainerV2_FLARE_Pseudo__nnUNetPlansFLARE24Big/nnUNetPlansFLARE24Big_plans_3D.pkl',
                'all', '/data1/userdisk0/zychen/RESULTS_FOLDER/nnUNet/3d_fullres/Task666_FLARE2024Challenge/'
                       'nnUNetTrainerV2_FLARE_Big__nnUNetPlansFLARE24Big',
                '/czy_SSD/1T/zychen/FLARE2024-Code/nnUNet_preprocessed/Task666_FLARE2024Challenge', True, 1, True, False, False)
data['name'] = 'nnUNetTrainerV2_FLARE_Pseudo'
data['plans']['modalities'] = {0: 'MR'}
data['plans']['normalization_schemes'] = OrderedDict([(0, 'MR')])
with open(root, 'wb') as file:
    pickle.dump(data, file)
