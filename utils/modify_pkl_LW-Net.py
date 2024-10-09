import pickle


### For LW-Net
# Before prediction, modify DDP to normal
root = '/data1/userdisk0/zychen/RESULTS_FOLDER/nnUNet/3d_fullres/' \
       'Task999_PseudoMR/nnUNetTrainerV2_FLARE_Small__nnUNetPlansFLARE24Small/all/model_final_checkpoint.model.pkl'
with open(root, 'rb') as file:
    data = pickle.load(file)
data['init'] = ('/data1/userdisk0/zychen/RESULTS_FOLDER/nnUNet/3d_fullres/Task999_PseudoMR/'
                'nnUNetTrainerV2__nnUNetPlansFLARE24Small/nnUNetPlansFLARE24Small_plans_3D.pkl',
                'all', '/data1/userdisk0/zychen/RESULTS_FOLDER/nnUNet/3d_fullres/Task999_PseudoMR/'
                       'nnUNetTrainerV2_FLARE_Small__nnUNetPlansFLARE24Small',
                '/erwen_SSD/2T/zychen/FLARE2024-Code/nnUNet_preprocessed/Task999_PseudoMR', True, 0, True, False, False)
data['name'] = 'nnUNetTrainerV2_FLARE_Small'
with open(root, 'wb') as file:
    pickle.dump(data, file)

