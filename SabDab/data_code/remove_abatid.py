import os
import os.path as osp
import shutil
from os.path import basename

datadir = '/data/gm_data/AbAtInteraction/AbAtIPA/abatInter_SCA/SabDab/SabDabdatabase'

def remove_abat_preEmbedDirs(datadir):
    pathls = [osp.join(datadir, pathi) for pathi in os.listdir(datadir) if (pathi.count('_') > 2) and osp.isdir(osp.join(datadir, pathi))]
    for path in pathls:
        if 'Abs' in os.listdir(path):
            shutil.rmtree(path)
    print('remove end in dir:', datadir)


if __name__ == "__main__":
    remove_abat_preEmbedDirs(datadir)
