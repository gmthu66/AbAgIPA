import esm
import torch
import numpy as np
import copy
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from antiberty import AntiBERTyRunner
from einops import repeat, rearrange
from utils.general import exists
from model.interface import SingleDataSamp, PhysicDataSamp
from database.batch_data import stack_attentions
from utils.abnumber_ import cdrDict2MaskTensor
from database.numbering import NUMberingAb
from spec_At.data_code.property_utils import PRO_property
ATOM_DIM = 3


class Embeder(object):
    def __init__(self, config, atbox_truc=15.0):
        self.device = torch.device('cpu') if 'device' not in dir(config) else config.device
        self.embed_type = config.embed_type if 'embed_type' in dir(config) else 'Bert'

        antiberty = AntiBERTyRunner(device=self.device)
        # self.antiberty = AntiBERTyRunner(device=torch.device('cpu'))
        antiberty.model.eval()
        if self.embed_type == ['Bert', 'Physic'][0]:
            esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.database_args = {'device': self.device, 'Ab_embed_model': antiberty, 'At_embed_model': esm_model, 'At_embed_alphabet': esm_alphabet, 'atbox_truc': atbox_truc}
        else:
            self.database_args = {'device': self.device, 'Ab_embed_model': antiberty, 'At_embed_model': None, 'At_embed_alphabet': None, 'atbox_truc': atbox_truc}


def append_sgls(data_ls, data, type=None, single_data=False):
    if isinstance(data, list):
        if len(data) == 1:
            if type == 'embeddings':  data = data[0].squeeze(0)
            data_ls.append(data)
        elif len(data) > 1:
            if type == 'embeddings':  data = torch.cat(data, dim=-2).squeeze(0)
            data_ls.append(data)  # 将heavy链和轻链合在一起
    else:
        data_ls.append(data)
    if not single_data:
        return data_ls
    else:
        return data


def get_singledata(ab_data, at_data, inter_label, pos_index=None, embed_type='Bert', cdr2mask=True, device_in=None, batch_size=1):
    if device_in is None:  device_in = torch.device('cpu')

    Ab_embeddings = append_sgls([], ab_data['Ab_embeddings'], single_data=True, type='embeddings').unsqueeze(0)
    Ab_VHLlens = [max(e.shape[1], 0) for e in ab_data['Ab_embeddings']]
    Ab_seqlen = sum(Ab_VHLlens)
    if cdr2mask:
        Ab_cdr_indexs = cdrDict2MaskTensor(ab_data['cdr_fr_dict'], seqlen=sum(Ab_VHLlens)).unsqueeze(0)

    if not exists(ab_data['Ab_temp_coords']) or (ab_data['Ab_temp_coords'] == []):
        Ab_temp_coords = torch.zeros(batch_size, 4 * Ab_seqlen, ATOM_DIM, device=device_in,).float()  # Ab的模板坐标被初始化为原点(4种原子)
    else:
        Ab_temp_coords = ab_data['Ab_temp_coords']
    if not exists(ab_data['Ab_temp_mask']) or (ab_data['Ab_temp_mask'] == []):
        Ab_temp_mask = torch.zeros(batch_size,
            4 * Ab_seqlen,).bool()
        if device_in is not None:  Ab_temp_mask = Ab_temp_mask.to(device_in)
    else:
        Ab_temp_mask = ab_data['Ab_temp_mask']
    if not exists(ab_data['Ab_batch_mask']):
        Ab_batch_mask = torch.ones(1, 4 * Ab_seqlen).bool()
        if device_in is not None:  Ab_batch_mask = Ab_batch_mask.to(device_in)
    else:
        Ab_batch_mask = ab_data['Ab_batch_mask']
    if not exists(ab_data['Ab_align_mask']):
        Ab_align_mask = torch.ones(batch_size,
            4 * Ab_seqlen,).bool()
        if device_in is not None:  Ab_align_mask = Ab_align_mask.to(device_in)
    else:
        Ab_align_mask = ab_data['Ab_align_mask']
    Ab_align_mask = Ab_align_mask & Ab_batch_mask  # Should already be masked by batch_mask anyway

    Ab_coords_label = ab_data['label_coords'].unsqueeze(0).to(device_in) if ab_data['label_coords'] is not None else None
    bert_attn_dim = ab_data['Ab_attentions'][0].size(1) * ab_data['Ab_attentions'][0].size(2)
    Ab_attentions = stack_attentions(ab_data['Ab_attentions'], Ab_VHLlens, bert_attn_dim)
    # one_data = SingleDataSamp()
    at_data = at_data[pos_index] if pos_index is not None else at_data
    At_coords_label = at_data['atbox_coords'][0].unsqueeze(0) if at_data is not None else None
    At_embeddings = at_data['At_embeddings'][0].unsqueeze(0) if at_data is not None else None
    At_attentions = at_data['chi_attentionsLs'][0].unsqueeze(0) if at_data is not None else None
    At_len = int(at_data['At_embeddings'][0].size(0)) if at_data is not None else None
    inter_label = torch.tensor(data=[inter_label], dtype=torch.long)
    At_batch_mask = torch.ones(1, 4 * At_len, device=device_in,).bool() if At_len is not None else At_len
    return_embeddings = True

    if embed_type is 'Bert':
        one_data = SingleDataSamp(
            Ab_embedding = Ab_embeddings.to(device_in),
            Ab_attention = Ab_attentions.to(device_in),
            Ab_seqlen = Ab_seqlen,
            Ab_VHLlen = Ab_VHLlens,
            Ab_maxlen = Ab_seqlen,
            Ab_temp_coords = Ab_temp_coords.to(device_in),
            Ab_temp_mask = Ab_temp_mask.to(device_in),
            Ab_batch_mask = Ab_batch_mask.to(device_in),
            Ab_align_mask = Ab_align_mask.to(device_in),
            Ab_coords_label = Ab_coords_label,
            Ab_cdr_indexs = Ab_cdr_indexs.bool().to(device_in),
            Ab_mut_range = ab_data['mutate_range'],
            return_embeddings = return_embeddings,
            At_embeddings= At_embeddings.to(device_in),
            At_coords_label=At_coords_label.to(device_in),
            At_attentions=At_attentions.to(device_in),
            At_batch_mask=At_batch_mask.to(device_in),
            At_len=At_len,
            inter_label=inter_label.to(device_in)
        )
    else:
        Ab_phisicEmbed = ab_data['Ab_phisicEmbed'][0].unsqueeze(0).to(device_in)
        Ab_phisicAttention = ab_data['Ab_phisicAttention'][0].unsqueeze(0).to(device_in)
        if At_embeddings is not None:
            t_list = [At_embeddings, At_coords_label, At_attentions, At_batch_mask] 
            At_embeddings, At_coords_label, At_attentions, At_batch_mask = map(lambda x: x.to(device_in), t_list)
        one_data = PhysicDataSamp(
            Ab_embedding = Ab_embeddings.to(device_in),
            Ab_phisicEmbed = Ab_phisicEmbed,
            Ab_attention = Ab_attentions.to(device_in),
            Ab_phisicAttention = Ab_phisicAttention,
            Ab_seqlen = Ab_seqlen,
            Ab_VHLlen = Ab_VHLlens,
            Ab_maxlen = Ab_seqlen,
            Ab_temp_coords = Ab_temp_coords.to(device_in),
            Ab_temp_mask = Ab_temp_mask.to(device_in),
            Ab_batch_mask = Ab_batch_mask.to(device_in),
            Ab_align_mask = Ab_align_mask.to(device_in),
            Ab_coords_label = Ab_coords_label,
            Ab_cdr_indexs = Ab_cdr_indexs.bool().to(device_in),
            Ab_mut_range = ab_data['mutate_range'],
            return_embeddings = return_embeddings,

            At_embeddings= At_embeddings,
            At_coords_label=At_coords_label,
            At_attentions=At_attentions,
            At_batch_mask=At_batch_mask,
            At_len=At_len,
            inter_label=inter_label.to(device_in)
        )
    return one_data


def collect_Ab_embeddings(ab_datas, knmae='Ab_embeddings'):
    Ab_embeddings, Ab_VHLlen_Ls, Ab_seqlens = [], [], []
    for ab_data in ab_datas:
        Ab_embeddings.append(append_sgls([], ab_data['Ab_embeddings'], single_data=True, type='embeddings'))
        Ab_VHLlens = [max(e.shape[1], 0) for e in ab_data['Ab_embeddings']]
        Ab_seqlen = sum(Ab_VHLlens)
        Ab_VHLlen_Ls.append(Ab_VHLlens)
        Ab_seqlens.append(Ab_seqlen)
    Ab_embeddings = pad_sequence(Ab_embeddings, batch_first=True, padding_value=0.0)
    return Ab_embeddings, Ab_VHLlen_Ls, Ab_seqlens


def collect_Ab_cdrMasks(ab_datas, kname='cdr_fr_dict', maxlen=None):
    # Ab_cdrMasks = torch.zeros((len(ab_datas), maxlen))
    Ab_cdrMasks = []
    for i_, abdata in enumerate(ab_datas):
        Ab_cdrMasks.append(cdrDict2MaskTensor(abdata[kname], maxlen))
    return torch.stack(Ab_cdrMasks, dim=0).bool()


def collect_Ab_temp_coords(ab_datas, maxlen, kname='Ab_temp_coords'):
    batch_size = len(ab_datas)
    Ab_temp_coords = torch.zeros(batch_size, maxlen * 4, 3)
    Ab_temp_masks = torch.zeros(batch_size, maxlen * 4)
    if not exists(ab_datas[0][kname]) or (ab_datas[0][kname] == []):
        pass
    else:
        for i_, abdata in enumerate(ab_datas):
            Ab_temp_coord = abdata[kname]
            Ab_temp_coords[i_, :Ab_temp_coords.size(1), :] = Ab_temp_coord
            Ab_temp_masks[i_, :Ab_temp_coords.size(1)] = 1.0
    return Ab_temp_coords, Ab_temp_masks


def collect_Ab_batch_mask(ab_datas, Ab_seqlens, kname='Ab_batch_mask', device=None):
    maxlen = max(Ab_seqlens)
    if not exists(ab_datas[0][kname]):
        if min(Ab_seqlens) == max(Ab_seqlens):
            Ab_batch_mask = torch.ones(len(ab_datas), 4 * maxlen).bool().to(device) if device is not None else torch.ones(len(ab_datas), 4 * maxlen).bool()
        else:
            Ab_batch_mask = torch.zeros(len(ab_datas), 4 * maxlen).bool().to(device) if device is not None else torch.zeros(len(ab_datas), 4 * maxlen).bool()
            for i, ab_data in enumerate(ab_datas):
                Ab_batch_maski = torch.ones(1, 4 * Ab_seqlens[i]).bool().to(device) if device is not None else torch.ones(1, 4 * Ab_seqlens[i]).bool()
                Ab_batch_mask[i:i+1, :Ab_batch_maski.size(1)] = Ab_batch_maski
                assert Ab_batch_maski.view(1, -1, 4).all(-1).sum() == Ab_seqlens[i]
                if ab_data['label_coords'] is not None:
                    assert ab_data['label_coords'].size(0) == Ab_seqlens[i]
    else:
        Ab_batch_mask = torch.zeros(len(ab_datas), 4 * maxlen).bool().to(device) if device is not None else torch.zeros(len(ab_datas), 4 * maxlen).bool()
        for i, ab_data in enumerate(ab_datas):
            Ab_batch_maski = ab_data[kname]
            Ab_batch_mask[i:i+1, :Ab_batch_maski.size(1)] = Ab_batch_maski
    return Ab_batch_mask


def collect_Ab_align_mask(ab_datas, Ab_seqlens, kname='Ab_align_mask', device=None):
    if device is None:  device = torch.device('cpu')
    maxlen = max(Ab_seqlens)
    if not exists(ab_datas[0][kname]):
        if min(Ab_seqlens) == maxlen:
            Ab_align_mask = torch.ones(len(ab_datas), 4 * maxlen, device=device,).bool()
        else:
            Ab_align_mask = torch.zeros(len(ab_datas), 4 * maxlen).bool()
            if device is not None:  Ab_align_mask = Ab_align_mask.to(device)
            for i, ab_data in enumerate(ab_datas):
                Ab_align_maski = torch.ones(1, 4 * Ab_seqlens[i]).bool().to(device) if device is not None else torch.ones(1, 4 * Ab_seqlens[i]).bool()
                Ab_align_mask[i:i+1, :Ab_align_maski.size(1)] = Ab_align_maski
    else:
        Ab_align_mask = torch.zeros(len(ab_datas), 4 * maxlen, device=device,)
        for i, ab_data in enumerate(ab_datas):
            Ab_align_maski = ab_data[kname]
            Ab_align_mask[i, Ab_align_maski.size(1)] = Ab_align_mask
    return Ab_align_mask


def collect_Ab_coords_label(ab_datas, maxlen, kname='label_coords', device=None):
    if not exists(ab_datas[0][kname]):
        Ab_coords_label = None
    else:
        Ab_coords_label = torch.zeros(len(ab_datas), maxlen, 5, 3).to(device) if device is not None else torch.zeros(len(ab_datas), maxlen, 5, 3)
        for i, abdata in enumerate(ab_datas):
            Ab_coords_label[i, :abdata[kname].size(0), :, :] = abdata[kname]
    return Ab_coords_label


def collect_Ab_attentions(ab_datas, maxlen, attnFeat_dim, Ab_VHLlens, kname='Ab_attentions', device=None):
    assert exists(ab_datas[0][kname])
    Ab_attentions = torch.zeros(len(ab_datas), maxlen, maxlen, attnFeat_dim)
    if device is not None:  Ab_attentions = Ab_attentions.to(device)
    for i, abdata in enumerate(ab_datas):
        Ab_attentioni = stack_attentions(abdata[kname], Ab_VHLlens[i], attnFeat_dim)
        Ab_attentions[i, :sum(Ab_VHLlens[i]), :sum(Ab_VHLlens[i]), :] = Ab_attentioni
    return Ab_attentions


def collect_At_coords_label(at_data, pos_index, batch_size, kname='atbox_coords'):
    if pos_index is None:
        at_data = at_data[0] if (isinstance(at_data, list) or isinstance(at_data, tuple)) else at_data
        At_coords_label = at_data[kname][0]
        At_coords_label = repeat(At_coords_label, "l a d -> b l a d", b=batch_size)
    else:
        At_coords_label = []
        for i_, posi in enumerate(pos_index):
            At_coords_labeli = rearrange(at_data[i_][kname][0], "l a d -> l (a d)", d=3)
            At_coords_label.append(At_coords_labeli)
        At_coords_label = pad_sequence(At_coords_label, batch_first=True, padding_value=0.0)
        At_coords_label = rearrange(At_coords_label, "b l (a d) -> b l a d", d=3)
    return At_coords_label


def collect_At_embeddings(at_data, pos_index, batch_size, kname='At_embeddings'):
    if pos_index is None:
        at_data = at_data[0] if (isinstance(at_data, list) or isinstance(at_data, tuple)) else at_data
        At_embeddingi = at_data[kname][0]
        At_len = [At_embeddingi.size(0)] * batch_size
        At_embeddings = repeat(At_embeddingi, "l d -> b l d", b=batch_size)
    else:
        At_embeddings, At_len = [], []
        for i_, posi in enumerate(pos_index):
            At_embeddingi = at_data[i_][kname][0]
            At_embeddings.append(At_embeddingi)
            At_len.append(At_embeddingi.size(0))
        At_embeddings = pad_sequence(At_embeddings, batch_first=True, padding_value=0.0)
    return At_embeddings, At_len


def collect_At_attentions(at_data, pos_index, batch_size, At_len=None, kname='chi_attentionsLs'):
    if pos_index is None:
        at_data = at_data[0] if (isinstance(at_data, list) or isinstance(at_data, tuple)) else at_data
        at_attns = at_data[kname][0]
        at_attns = repeat(at_attns, "l1 l2 d -> b l1 l2 d", b=batch_size)
    else:
        max_atlen, attn_dim = max(At_len), at_data[0][kname][0].size(-1)
        at_attns = torch.zeros(batch_size, max_atlen, max_atlen, attn_dim)
        for i_, atlen in enumerate(At_len):
            at_attns[i_, :atlen, :atlen, :] = at_data[i_][kname][0]
    return at_attns


def batch_mask2d_bylens(lens, atom=1):
    mask_tensor = torch.zeros((len(lens), max(lens) * atom), dtype=torch.long)
    for i, seqlen in enumerate(lens):
        mask_tensor[i, :seqlen * atom] = 1
    return mask_tensor.bool()


def collect_Ab_phisicEmbed(ab_data, kname='Ab_phisicEmbed', device=torch.device('cpu')):
    Ab_phisicEmbed = []
    for i, abdatai in enumerate(ab_data):
        Ab_phisicEmbedi = ab_data[i][kname][0]
        Ab_phisicEmbed.append(Ab_phisicEmbedi)
    Ab_phisicEmbed = pad_sequence(Ab_phisicEmbed, batch_first=True, padding_value=0.0)
    if device is not None:  Ab_phisicEmbed = Ab_phisicEmbed.to(device)
    return Ab_phisicEmbed


def collect_Ab_phisicAttention(ab_data, maxab_len, attn_dim, kname='Ab_phisicAttention', device=torch.device('cpu')):
    Ab_phisicAttention = torch.zeros(len(ab_data), maxab_len, maxab_len, attn_dim, dtype=torch.float32, device=device)
    for i, abdatai in enumerate(ab_data):
        ab_physic_attni = abdatai[kname][0]
        Ab_phisicAttention[i, :ab_physic_attni.size(0), :ab_physic_attni.size(1), :] = ab_physic_attni

    if device is not None:  Ab_phisicAttention = Ab_phisicAttention.to(device)
    return Ab_phisicAttention


def collect_listelem(datas, kname):
    elemls = []
    for datai in datas:
        if datai[kname] is None:
            return None
        elemls.append(datai[kname])
    return elemls


def get_batchdata(ab_data, at_data, inter_label, pos_index=None, embed_type='Bert', cdr2mask=True, device_in=None, batch_size=1):
    if batch_size > 1:
        device_in = device_in[0] if device_in is not None else None
        Ab_embeddings, Ab_VHLlens, Ab_seqlen = collect_Ab_embeddings(ab_data)
        if cdr2mask:
            Ab_cdr_indexs = collect_Ab_cdrMasks(ab_data, maxlen=max(Ab_seqlen))

        Ab_temp_coords, Ab_temp_mask = collect_Ab_temp_coords(ab_data, maxlen=max(Ab_seqlen))
        Ab_temp_coords = Ab_temp_coords.float().to(device_in) if device_in is not None else Ab_temp_coords.float()
        Ab_temp_mask = Ab_temp_mask.bool().to(device_in) if device_in is not None else Ab_temp_mask.bool()
        Ab_maxlen = max(Ab_seqlen)

        Ab_batch_mask = collect_Ab_batch_mask(ab_data, Ab_seqlen, device=device_in)  # 在获取Ab_bact_mask时报错了
        Ab_align_mask = collect_Ab_align_mask(ab_data, Ab_seqlen, device=device_in)
    else:
        Ab_embeddings = append_sgls([], ab_data['Ab_embeddings'], single_data=True, type='embeddings').unsqueeze(0)
        Ab_VHLlens = [max(e.shape[1], 0) for e in ab_data['Ab_embeddings']]
        Ab_seqlen, Ab_maxlen = sum(Ab_VHLlens), sum(Ab_VHLlens)
        if not exists(ab_data['Ab_temp_coords']) or (ab_data['Ab_temp_coords'] == []):
            Ab_temp_coords = torch.zeros(batch_size, 4 * Ab_seqlen, ATOM_DIM, device=device_in,).float()  # Ab的模板坐标被初始化为原点(4种原子)
        else:
            Ab_temp_coords = ab_data['Ab_temp_coords']
        if cdr2mask:
            Ab_cdr_indexs = cdrDict2MaskTensor(ab_data['cdr_fr_dict'], seqlen=sum(Ab_VHLlens)).unsqueeze(0)

        if not exists(ab_data['Ab_temp_mask']) or (ab_data['Ab_temp_mask'] == []):
            Ab_temp_mask = torch.zeros(batch_size, 4 * Ab_seqlen, device=device_in,).bool()
        else:
            Ab_temp_mask = ab_data['Ab_temp_mask']
        if not exists(ab_data['Ab_batch_mask']):
            Ab_batch_mask = torch.ones(1, 4 * Ab_seqlen).bool().to(device_in)
        else:
            Ab_batch_mask = ab_data['Ab_batch_mask']
        if not exists(ab_data['Ab_align_mask']):
            Ab_align_mask = torch.ones(batch_size, 4 * Ab_seqlen, device=device_in,).bool()
        else:
            Ab_align_mask = ab_data['Ab_align_mask']
    Ab_align_mask = Ab_align_mask & Ab_batch_mask  # Should already be masked by batch_mask anyway

    if batch_size > 1:
        Ab_coords_label = collect_Ab_coords_label(ab_data, max(Ab_seqlen), kname='label_coords', device=device_in)
        bert_attn_dim = ab_data[0]['Ab_attentions'][0].size(1) * ab_data[0]['Ab_attentions'][0].size(2)
        Ab_attentions = collect_Ab_attentions(ab_data, max(Ab_seqlen), bert_attn_dim, Ab_VHLlens, kname='Ab_attentions', device=None)
    else:
        Ab_coords_label = ab_data['label_coords'].unsqueeze(0).to(device_in) if ab_data['label_coords'] is not None else None
        Ab_attentions = stack_attentions(ab_data['Ab_attentions'], Ab_VHLlens, bert_attn_dim)

    if batch_size > 1:
        if embed_type is not 'Bert':
            Ab_phisicEmbed = collect_Ab_phisicEmbed(ab_data, kname='Ab_phisicEmbed', device=device_in)
            attndim = ab_data[0]['Ab_phisicAttention'][0].size(-1)
            Ab_phisicAttention = collect_Ab_phisicAttention(ab_data, max(Ab_seqlen), attndim, kname='Ab_phisicAttention', device=device_in)
        Ab_mut_range = collect_listelem(ab_data, kname='mutate_range')
    else:
        if embed_type is not 'Bert':
            Ab_phisicEmbed = ab_data['Ab_phisicEmbed'][0].unsqueeze(0).to(device_in)
            Ab_phisicAttention = ab_data['Ab_phisicAttention'][0].unsqueeze(0).to(device_in)
        Ab_mut_range = ab_data['mutate_range']

    # -------------------------- 获得at数据------------------------
    if batch_size > 1:
        At_coords_label = collect_At_coords_label(at_data, pos_index, batch_size, kname='atbox_coords')
        At_embeddings, At_len = collect_At_embeddings(at_data, pos_index, batch_size, kname='At_embeddings')
        At_attentions = collect_At_attentions(at_data, pos_index, batch_size, At_len=At_len, kname='chi_attentionsLs')
        inter_label = torch.tensor(data=[list(inter_label)], dtype=torch.long, device=At_embeddings[0].device)
        At_batch_mask = batch_mask2d_bylens(At_len, atom=4).to(device_in)    
    else:
        at_data = at_data[pos_index] if pos_index is not None else at_data  # pos_index为None, 则直接对at_data进行stack
        At_coords_label = at_data['atbox_coords'][0].unsqueeze(0)
        At_embeddings = at_data['At_embeddings'][0].unsqueeze(0)
        At_attentions, At_len = at_data['chi_attentionsLs'][0].unsqueeze(0), int(at_data['At_embeddings'][0].size(0))
        inter_label = torch.tensor(data=[inter_label], dtype=torch.long, device=At_embeddings[0].device)
        At_batch_mask = torch.ones(1, 4 * At_len, device=device_in,).bool()
    return_embeddings = True

    if embed_type is 'Bert':
        one_data = SingleDataSamp(
            Ab_embedding = Ab_embeddings.to(device_in),
            Ab_attention = Ab_attentions.to(device_in),
            Ab_seqlen = Ab_seqlen,
            Ab_VHLlen = Ab_VHLlens,
            Ab_maxlen = Ab_maxlen,
            Ab_temp_coords = Ab_temp_coords.to(device_in),
            Ab_temp_mask = Ab_temp_mask.to(device_in),
            Ab_batch_mask = Ab_batch_mask.to(device_in),
            Ab_align_mask = Ab_align_mask.to(device_in),
            Ab_coords_label = Ab_coords_label,
            Ab_cdr_indexs = Ab_cdr_indexs.bool().to(device_in),
            Ab_mut_range = Ab_mut_range,
            return_embeddings = return_embeddings,
            At_embeddings= At_embeddings.to(device_in),
            At_coords_label=At_coords_label.to(device_in),
            At_attentions=At_attentions.to(device_in),
            At_batch_mask=At_batch_mask.to(device_in),
            At_len=At_len,
            inter_label=inter_label.to(device_in)
        )
    else:
        one_data = PhysicDataSamp(
            Ab_embedding = Ab_embeddings.to(device_in),
            Ab_phisicEmbed = Ab_phisicEmbed,
            Ab_attention = Ab_attentions.to(device_in),
            Ab_phisicAttention = Ab_phisicAttention,
            Ab_seqlen = Ab_seqlen,
            Ab_VHLlen = Ab_VHLlens,
            Ab_maxlen = Ab_maxlen,
            Ab_temp_coords = Ab_temp_coords.to(device_in),
            Ab_temp_mask = Ab_temp_mask.to(device_in),
            Ab_batch_mask = Ab_batch_mask.to(device_in),
            Ab_align_mask = Ab_align_mask.to(device_in),
            Ab_coords_label = Ab_coords_label,
            Ab_cdr_indexs = Ab_cdr_indexs.bool().to(device_in),
            Ab_mut_range = Ab_mut_range,
            return_embeddings = return_embeddings,

            At_embeddings= At_embeddings.to(device_in),
            At_coords_label=At_coords_label.to(device_in),
            At_attentions=At_attentions.to(device_in),
            At_batch_mask=At_batch_mask.to(device_in),
            At_len=At_len,
            inter_label=inter_label.to(device_in)
        )
    return one_data
# def to_device(obj_Data, device_in):
#     for attr_name in dir(obj_Data):
#         if '_' not in attr_name:
#             attr = getattr(obj_Data, attr_name)
#             if isinstance(attr, torch.Tensor):
#                 setattr(obj_Data, attr_name, attr.to(device=device_in))
#     return obj_Data


class GetAAPhysic(object):
    def __init__(self):
        self.pos_encode = True

    def _get_aaseq_featureM(self, aa_seq, ab_flag=False, uniform_flg=False, frcdr_dict=None):
        self.abnumber = NUMberingAb(rule='chothia')
        """特征处理函数: 注意数据都是torch Tensor类型\n
        输入是氨基酸字符串--一级序列  返回one-hot向量构成的矩阵\n
        返回范德华半径、 分子量、 侧链类型(6种)、 等电点、  疏水性值
        ab_flag: 指定序列是否是抗体序列\n"""
        self.numbered_ab_flag = False
        alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
        aa_seq_ini = copy.deepcopy(aa_seq)
        if ab_flag and self.pos_encode and (frcdr_dict is None):  # 用来指定是否进行位置编码, 且位置编码只针对abseq进行
            self.numbered_ab_flag = self.abnumber.numberingNb(aa_seq_ini)
        else:
            pass
        aa_seq_ini_int = np.array(list(aa_seq), dtype='|S1').view(np.uint8)
        aa_seq = copy.deepcopy(aa_seq_ini_int)
        prop = PRO_property(uniform=uniform_flg)  # j
        dict_sidetype = prop.dict_sidetype  # one-hot类型
        dict_kdhydro = prop.dict_kdhydro
        dict_iep = prop.dict_iep
        dict_vdW = prop.dict_vdW
        dict_mol_weight = prop.dict_mol_weight
        # aa_seq_forprop = copy.deepcopy(aa_seq_ini_int)
        aa_seq_sidetype = copy.deepcopy(aa_seq_ini_int)
        aa_seq_iep = copy.deepcopy(aa_seq_ini_int).astype(np.float)
        aa_seq_vdW = copy.deepcopy(aa_seq_ini_int).astype(np.float)
        aa_seq_molw = copy.deepcopy(aa_seq_ini_int).astype(np.float)
        aa_seq_kdhydro = copy.deepcopy(aa_seq_ini_int).astype(np.float)
        for i_ in range(alphabet.shape[0]):
            chr_str = chr(alphabet[i_])
            aa_seq[aa_seq == alphabet[i_]] = i_
            aa_seq_sidetype[aa_seq_ini_int == alphabet[i_]] = dict_sidetype[chr_str]
            aa_seq_iep[aa_seq_ini_int == alphabet[i_]] = dict_iep[chr_str]
            aa_seq_molw[aa_seq_ini_int == alphabet[i_]] = dict_mol_weight[chr_str]
            aa_seq_vdW[aa_seq_ini_int == alphabet[i_]] = dict_vdW[chr_str]
            aa_seq_kdhydro[aa_seq_ini_int == alphabet[i_]] = dict_kdhydro[chr_str]
        aa_seq[aa_seq > 20] = 20
        nbin = 21  # nbin或许应该当改为21？
        aa_seq = torch.tensor(data=aa_seq, dtype=torch.int64)
        aa_seq_onehot = F.one_hot(aa_seq, num_classes=nbin).to(torch.float32)
        aa_seq_sidetype = torch.tensor(data=aa_seq_sidetype, dtype=torch.int64)
        aa_seq_sidetype_onehot = F.one_hot(aa_seq_sidetype, num_classes=len(set(dict_sidetype.values())))
        aa_seq_sidetype_onehot = aa_seq_sidetype_onehot.to(torch.float32)
        aa_seq_iep = torch.tensor(data=aa_seq_iep, dtype=torch.float32).unsqueeze(1)
        aa_seq_molw = torch.tensor(data=aa_seq_molw, dtype=torch.float32).unsqueeze(1)
        aa_seq_vdW = torch.tensor(data=aa_seq_vdW, dtype=torch.float32).unsqueeze(1)
        aa_seq_kdhydro = torch.tensor(data=aa_seq_kdhydro, dtype=torch.float32).unsqueeze(1)
        msa_mat = torch.cat((aa_seq_onehot, aa_seq_iep, aa_seq_kdhydro, aa_seq_molw, aa_seq_vdW, aa_seq_sidetype_onehot), axis=1)  # 各个氨基酸的类型
        # 为Ab序列添加编码Numbering信息, 分为fr cdr的index_list进行添加
        if frcdr_dict is not None:
            cdr_index, tail_index = frcdr_dict['cdr_range'].numpy(), np.array([])
            fr_index = np.array(list(set(list(range(aa_seq.size(0)))) - set(cdr_index)))
            tail_flag, numbered_ab_flag = False, True
        elif ab_flag:
            # self.abnumber.tail_index = np.array(list(set(list(range(aa_seq.size(0)))) - set(list(range(frcdr_dict['max_cdrfr_index'] + 1)))))
            # self.abnumber.cdr_index = frcdr_dict['cdr_range'].numpy()
            # self.abnumber.fr_index = np.array(list(set(list(range(aa_seq.size(0)))) - set(list(self.abnumber.tail_index) + list(self.abnumber.cdr_index.numpy()))))
            # if len(self.abnumber.tail_index) > 0:
            #     self.abnumber.tail_flag = True
            cdr_index, tail_index = self.abnumber.cdr_index, self.abnumber.tail_index
            fr_index = self.abnumber.fr_index
            tail_flag, numbered_ab_flag = self.abnumber.tail_flag, self.numbered_ab_flag
        if ab_flag and numbered_ab_flag:  # self.numbered_ab会添加到physic特征部分
            nubered_mat = torch.zeros(msa_mat.size(0), 4)
            for x_ in fr_index:  # 为fr区域添加标签
                nubered_mat[x_, :] = torch.tensor(data=np.array([1, 0, 0, 0]), dtype=torch.float32)
            for x_ in cdr_index:  # 为cdr区域添加标签
                nubered_mat[x_, :] = torch.tensor(data=np.array([0, 1, 0, 0]), dtype=torch.float32)
            if tail_flag:
                for x_ in tail_index:
                    nubered_mat[x_, :] = torch.tensor(data=np.array([0, 0, 1, 0]), dtype=torch.float32)
            msa_mat = torch.cat((nubered_mat, msa_mat), axis=1)
            if frcdr_dict is None:
                assert len(aa_seq_ini) == (len(self.abnumber.tail_numbered_idls) + len(self.abnumber.seq_numbered_idls))
        elif ab_flag and not numbered_ab_flag:
            nubered_mat = torch.tensor(data=np.array([0, 0, 0, 1]), dtype=torch.float32).repeat(msa_mat.size(0)).reshape(-1, 4)
            msa_mat = torch.cat((nubered_mat, msa_mat), axis=1)
        else:  # 为非抗原数据, 则在开头添加[0, 0, 0, 1]编码
            nubered_mat = torch.tensor(data=np.array([0, 0, 0, 1]), dtype=torch.float32).repeat(msa_mat.size(0)).reshape(-1, 4)
            msa_mat = torch.cat((nubered_mat, msa_mat), axis=1)
        return msa_mat
