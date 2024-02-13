import os.path as osp
from tqdm import tqdm
from spec_At.utils_df import string_to_dict
from spec_At.Covid19.get_Covid_data import process_Abdata


def pre_embed(train_val_test_dict, params=None, abtype='not_mutante'):
    pre_embed_dict = {}

    init_pos_data = params['init_pos_data']
    embeder = params['embeder']
    template_flg = params['template_flg']
    for k_ in train_val_test_dict.keys():
        data_df = train_val_test_dict[k_]
        pre_embed_dict[k_] = {}
        print(f'pre_embeding... {k_}')

        for idx in tqdm(list(data_df.index)):
            data_si = data_df.loc[idx]
            aaseq_dict = get_aaseqDict_BySeries(data_si)
            mutate_range = eval(data_si['MutRange']) if 'MutRange' in data_si.index else None
            cdr_fr_info = string_to_dict(data_si['cdr_fr_info'])
            pos_index = int(data_si['pos_dataid'])
            Ab_data = process_Abdata(aaseq_dict, wild_Abdata_dict=init_pos_data['Ab_data'], pos_index=pos_index, rec_cdrfr=cdr_fr_info, db_args=embeder.database_args, tmp_flg=template_flg, mutate_range=mutate_range, map_cpu=True)
            data_id = int(data_si['data_id'])
            pre_embed_dict[k_][data_id] = Ab_data
            # print()
    print(f'pre_embeding... end')
    return pre_embed_dict


def get_aaseqDict_BySeries(dfi, hseq_name='Hseq', lseq_name='Lseq'):
    aaseq_dict = {}
    if hseq_name in dfi:
        if isinstance(dfi[hseq_name], str):  aaseq_dict['H'] = dfi[hseq_name]
    if lseq_name in dfi:
        if isinstance(dfi[lseq_name], str):  aaseq_dict['L'] = dfi[lseq_name]
    return aaseq_dict
