import os
import shutil

root = '/data1/userdisk0/zychen/flare2024/registered-1_new'
save = '/data1/userdisk0/zychen/flare2024/imagesTr/Register_MR'

patients = os.listdir(root)
print(len(patients))
for patient in patients:
    phases = os.listdir(os.path.join(root, patient))
    for phase in phases:
        new_name = patient + '_' + phase[:-7].replace(" ", "") + '_0000.nii.gz'
        print(new_name)
        shutil.copy(os.path.join(root, patient, phase), os.path.join(save, new_name))
