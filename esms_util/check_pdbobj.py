import os
import os.path as osp


pt_ls = os.listdir('/data/gm_data/AbAtInteraction/AbAtIPA/abatInter_SCA/esms_util/Abs/pre_embed')
pdbobj_ls = [pt.split('_')[0] for pt in pt_ls]
print()
