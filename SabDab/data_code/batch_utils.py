import esm
import torch
from torch.nn.utils.rnn import pad_sequence
from antiberty import AntiBERTyRunner
from einops import repeat, rearrange
from utils.general import exists
from model.interface import SingleDataSamp, PhysicDataSamp
from database.batch_data import stack_attentions
from utils.abnumber_ import cdrDict2MaskTensor
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
            4 * Ab_seqlen,
            device=device_in,).bool()
    else:
        Ab_temp_mask = ab_data['Ab_temp_mask']
    if not exists(ab_data['Ab_batch_mask']):
        Ab_batch_mask = torch.ones(1, 4 * Ab_seqlen).bool().to(device_in)
    else:
        Ab_batch_mask = ab_data['Ab_batch_mask']
    if not exists(ab_data['Ab_align_mask']):
        Ab_align_mask = torch.ones(batch_size,
            4 * Ab_seqlen,
            device=device_in,).bool()
    else:
        Ab_align_mask = ab_data['Ab_align_mask']
    Ab_align_mask = Ab_align_mask & Ab_batch_mask  # Should already be masked by batch_mask anyway

    Ab_coords_label = ab_data['label_coords'].unsqueeze(0).to(device_in) if ab_data['label_coords'] is not None else None
    bert_attn_dim = ab_data['Ab_attentions'][0].size(1) * ab_data['Ab_attentions'][0].size(2)
    Ab_attentions = stack_attentions(ab_data['Ab_attentions'], Ab_VHLlens, bert_attn_dim)
    # one_data = SingleDataSamp()
    at_data = at_data[pos_index] if pos_index is not None else at_data
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

            At_embeddings= At_embeddings.to(device_in),
            At_coords_label=At_coords_label.to(device_in),
            At_attentions=At_attentions.to(device_in),
            At_batch_mask=At_batch_mask.to(device_in),
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
    maxlen, not_exists = max(Ab_seqlens), False
    if kname not in ab_datas[0]:  not_exists = True
    else:
        if not exists(ab_datas[0][kname]):  not_exists = True  # 这是对于Ab等长时的情况, 但是此时是不可以的
    if not_exists:
        if min(Ab_seqlens) == max(Ab_seqlens):
            Ab_batch_mask = torch.ones(len(ab_datas), 4 * maxlen).bool().to(device) if device is not None else torch.ones(len(ab_datas), 4 * maxlen).bool()
        else:
            Ab_batch_mask = torch.zeros(len(ab_datas), 4 * maxlen).bool().to(device) if device is not None else torch.zeros(len(ab_datas), 4 * maxlen).bool()
            for i, ab_data in enumerate(ab_datas):
                Ab_batch_maski = torch.ones(1, 4 * Ab_seqlens[i]).bool().to(device) if device is not None else torch.ones(1, 4 * Ab_seqlens[i]).bool()
                Ab_batch_mask[i:i+1, :Ab_batch_maski.size(1)] = Ab_batch_maski
                assert Ab_batch_maski.view(1, -1, 4).all(-1).sum() == Ab_seqlens[i]
                if ab_data['label_coords'] is not None:  assert ab_data['label_coords'].size(0) == Ab_seqlens[i]
    else:
        Ab_batch_mask = torch.zeros(len(ab_datas), 4 * maxlen).bool().to(device) if device is not None else torch.zeros(len(ab_datas), 4 * maxlen).bool()
        for i, ab_data in enumerate(ab_datas):
            Ab_batch_maski = ab_data[kname]
            Ab_batch_mask[i:i+1, :Ab_batch_maski.size(1)] = Ab_batch_maski
    return Ab_batch_mask


def collect_Ab_align_mask(ab_datas, Ab_seqlens, kname='Ab_align_mask', device=None):
    if device is None:  device = torch.device('cpu')
    maxlen, not_exists = max(Ab_seqlens), False
    if kname not in ab_datas[0]:  not_exists = True
    else:
        if not exists(ab_datas[0][kname]):  not_exists = True  # 这是对于Ab等长时的情况, 但是此时是不可以的
    if not_exists:
        if min(Ab_seqlens) == maxlen:
            Ab_align_mask = torch.ones(len(ab_datas), 4 * maxlen, device=device,).bool()
        else:
            Ab_align_mask = torch.zeros(len(ab_datas), 4 * maxlen).bool()
            if device is not None:  Ab_align_mask = Ab_align_mask.to(device)
            for i, ab_data in enumerate(ab_datas):
                Ab_align_maski = torch.ones(1, 4 * Ab_seqlens[i]).bool()
                if device is not None:  Ab_align_maski = Ab_align_maski.to(device)
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
            At_coords_labeli = at_data[i_][kname][0] if isinstance(at_data[i_][kname][0], list) else at_data[i_][kname]
            At_coords_labeli = rearrange(At_coords_labeli, "l a d -> l (a d)", d=3)
            At_coords_label.append(At_coords_labeli)
        At_coords_label = pad_sequence(At_coords_label, batch_first=True, padding_value=0.0)
        At_coords_label = rearrange(At_coords_label, "b l (a d) -> b l a d", d=3)
    return At_coords_label


def collect_At_embeddings(at_data, pos_index, batch_size, kname='At_embeddings'):
    if pos_index is None:
        at_data = at_data[0] if (isinstance(at_data, list) or isinstance(at_data, tuple)) else at_data
        At_embeddingi = at_data[kname][0] if isinstance(at_data[kname], list) else at_data[kname]
        At_len = [At_embeddingi.size(0)] * batch_size
        At_embeddings = repeat(At_embeddingi, "l d -> b l d", b=batch_size)
    else:
        At_embeddings, At_len = [], []
        for i_, posi in enumerate(pos_index):
            At_embeddingi = at_data[i_][kname][0] if isinstance(at_data[i_][kname], list) else at_data[i_][kname]
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
            at_attni = at_data[i_][kname][0] if isinstance(at_data[i_][kname], list) else at_data[i_][kname]
            at_attns[i_, :atlen, :atlen, :] = at_attni
    return at_attns


def batch_mask2d_bylens(lens, atom=1):
    mask_tensor = torch.zeros((len(lens), max(lens) * atom), dtype=torch.long)
    for i, seqlen in enumerate(lens):
        mask_tensor[i, :seqlen * atom] = 1
    return mask_tensor.bool()


def collect_Ab_phisicEmbed(ab_data, kname='Ab_phisicEmbed', device=None):
    Ab_phisicEmbed = []
    for i, abdatai in enumerate(ab_data):
        Ab_phisicEmbedi = ab_data[i][kname][0] if isinstance(ab_data[i][kname], list) else ab_data[i][kname]
        Ab_phisicEmbed.append(Ab_phisicEmbedi)
    Ab_phisicEmbed = pad_sequence(Ab_phisicEmbed, batch_first=True, padding_value=0.0)
    Ab_phisicEmbed = Ab_phisicEmbed
    if device is not None:  Ab_phisicEmbed = Ab_phisicEmbed.to(device)
    return Ab_phisicEmbed


def collect_Ab_phisicAttention(ab_data, maxab_len, attn_dim, kname='Ab_phisicAttention', device=None):
    Ab_phisicAttention = torch.zeros(len(ab_data), maxab_len, maxab_len, attn_dim, dtype=torch.float32)
    for i, abdatai in enumerate(ab_data):
        ab_physic_attni = abdatai[kname][0] if isinstance(abdatai[kname], list) else abdatai[kname]
        Ab_phisicAttention[i, :ab_physic_attni.size(0), :ab_physic_attni.size(1), :] = ab_physic_attni

    if device is not None:  Ab_phisicAttention = Ab_phisicAttention.to(device)
    return Ab_phisicAttention


def collect_listelem(datas, kname):
    elemls = []
    for datai in datas:
        if kname not in datai.keys():  return None
        if datai[kname] is None:
            return None
        elemls.append(datai[kname])
    return elemls


def collect_predCoords(ab_data, kname='pred_coords'):
    Ab_pred_coodsls, Ab_VHLlens, Ab_seqlen = [], [], []
    for i_, abdatai in enumerate(ab_data):
        Ab_pred_coords = abdatai[kname].squeeze()
        Ab_pred_coodsls.append(Ab_pred_coords)

        Ab_seqlen.append(Ab_pred_coords.size(0))
        Hlen = len(abdatai['seq_dict']['H']) if 'H' in abdatai['seq_dict'] else 0
        Llen = len(abdatai['seq_dict']['L']) if 'L' in abdatai['seq_dict'] else 0
        Ab_VHLlens.append([Hlen, Llen])
    Ab_pred_coodsls = pad_sequence(Ab_pred_coodsls, batch_first=True, padding_value=0.0)
    return Ab_pred_coodsls, Ab_VHLlens, Ab_seqlen


def get_batchdata(ab_data, at_data, inter_label, pos_index=None, embed_type='Bert', cdr2mask=True, device_in=None, batch_size=1, Ab_abatid=None):
    Ab_embeddings, Ab_attentions, Ab_temp_coords, Ab_temp_mask, Ab_coords_label, Ab_predCooords = None, None, None, None, None, None
    # if batch_size > 1:
    device_in = device_in[0] if device_in is not None else None
    if 'pred_coords' in ab_data[0].keys():
        Ab_predCooords, Ab_VHLlens, Ab_seqlen = collect_predCoords(ab_data)

    if Ab_predCooords is None:
        Ab_embeddings, Ab_VHLlens, Ab_seqlen = collect_Ab_embeddings(ab_data)
        Ab_temp_coords, Ab_temp_mask = collect_Ab_temp_coords(ab_data, maxlen=max(Ab_seqlen))
        Ab_temp_coords = Ab_temp_coords.float().to(device_in) if device_in is not None else Ab_temp_coords.float() 
        Ab_temp_mask = Ab_temp_mask.bool().to(device_in) if device_in is not None else Ab_temp_mask.bool()

    if cdr2mask:
        Ab_cdr_indexs = collect_Ab_cdrMasks(ab_data, maxlen=max(Ab_seqlen))  # 只收集Antibody的CDR区域
    Ab_maxlen = max(Ab_seqlen)

    Ab_batch_mask = collect_Ab_batch_mask(ab_data, Ab_seqlen, device=device_in)  # 在获取Ab_bact_mask时报错了
    Ab_align_mask = collect_Ab_align_mask(ab_data, Ab_seqlen, device=device_in)
    # else:
    #     Ab_embeddings = append_sgls([], ab_data['Ab_embeddings'], single_data=True, type='embeddings').unsqueeze(0)
    #     Ab_VHLlens = [max(e.shape[1], 0) for e in ab_data['Ab_embeddings']]
    #     Ab_seqlen, Ab_maxlen = sum(Ab_VHLlens), sum(Ab_VHLlens)
    #     if not exists(ab_data['Ab_temp_coords']) or (ab_data['Ab_temp_coords'] == []):
    #         Ab_temp_coords = torch.zeros(batch_size, 4 * Ab_seqlen, ATOM_DIM).float()  # Ab的模板坐标被初始化为原点(4种原子)
    #     else:
    #         Ab_temp_coords = ab_data['Ab_temp_coords']
    #     if cdr2mask:
    #         Ab_cdr_indexs = cdrDict2MaskTensor(ab_data['cdr_fr_dict'], seqlen=sum(Ab_VHLlens)).unsqueeze(0)

    #     if not exists(ab_data['Ab_temp_mask']) or (ab_data['Ab_temp_mask'] == []):
    #         Ab_temp_mask = torch.zeros(batch_size, 4 * Ab_seqlen).bool()
    #     else:
    #         Ab_temp_mask = ab_data['Ab_temp_mask']
    #     if not exists(ab_data['Ab_batch_mask']):
    #         Ab_batch_mask = torch.ones(1, 4 * Ab_seqlen).bool()
    #     else:
    #         Ab_batch_mask = ab_data['Ab_batch_mask']
    #     if not exists(ab_data['Ab_align_mask']):
    #         Ab_align_mask = torch.ones(batch_size, 4 * Ab_seqlen).bool()
    #     else:
    #         Ab_align_mask = ab_data['Ab_align_mask']
    Ab_align_mask = Ab_align_mask & Ab_batch_mask  # Should already be masked by batch_mask anyway

    # if batch_size > 1:
    if Ab_predCooords is None:
        Ab_coords_label = collect_Ab_coords_label(ab_data, max(Ab_seqlen), kname='label_coords', device=device_in)
        bert_attn_dim = ab_data[0]['Ab_attentions'][0].size(1) * ab_data[0]['Ab_attentions'][0].size(2)
        Ab_attentions = collect_Ab_attentions(ab_data, max(Ab_seqlen), bert_attn_dim, Ab_VHLlens, kname='Ab_attentions', device=None)
    # else:
    #     Ab_coords_label = ab_data['label_coords'].unsqueeze(0).to(device_in) if ab_data['label_coords'] is not None else None
    #     Ab_attentions = stack_attentions(ab_data['Ab_attentions'], Ab_VHLlens, bert_attn_dim)

    # if batch_size > 1:
    if embed_type is not 'Bert':
        Ab_phisicEmbed = collect_Ab_phisicEmbed(ab_data, kname='Ab_phisicEmbed', device=device_in)
        attndim = ab_data[0]['Ab_phisicAttention'][0].size(-1)
        Ab_phisicAttention = collect_Ab_phisicAttention(ab_data, max(Ab_seqlen), attndim, kname='Ab_phisicAttention', device=device_in)
    Ab_mut_range = collect_listelem(ab_data, kname='mutate_range')
    # # else:
    #     if embed_type is not 'Bert':
    #         Ab_phisicEmbed = ab_data['Ab_phisicEmbed'][0].unsqueeze(0)
    #         Ab_phisicAttention = ab_data['Ab_phisicAttention'][0].unsqueeze(0)
    #     Ab_mut_range = ab_data['mutate_range']

    # -------------------------- 获得at数据------------------------
    #  if batch_size > 1:
    At_coords_label = collect_At_coords_label(at_data, pos_index, batch_size, kname='atbox_coords')
    At_embeddings, At_len = collect_At_embeddings(at_data, pos_index, batch_size, kname='At_embeddings')
    At_attentions = collect_At_attentions(at_data, pos_index, batch_size, At_len=At_len, kname='chi_attentionsLs')
    inter_label = torch.tensor(data=[list(inter_label)], dtype=torch.long)
    At_batch_mask = batch_mask2d_bylens(At_len, atom=4)
    # else:
    #     at_data = at_data[pos_index] if pos_index is not None else at_data  # pos_index为None, 则直接对at_data进行stack
    #     At_coords_label = at_data['atbox_coords'][0].unsqueeze(0)
    #     At_embeddings = at_data['At_embeddings'][0].unsqueeze(0)
    #     At_attentions, At_len = at_data['chi_attentionsLs'][0].unsqueeze(0), int(at_data['At_embeddings'][0].size(0))
    #     inter_label = torch.tensor(data=[inter_label], dtype=torch.long)
    #     At_batch_mask = torch.ones(1, 4 * At_len).bool()
    return_embeddings = True

    if embed_type is 'Bert':
        one_data = SingleDataSamp(
            Ab_embedding = Ab_embeddings,
            Ab_attention = Ab_attentions,
            Ab_seqlen = Ab_seqlen,
            Ab_VHLlen = Ab_VHLlens,
            Ab_maxlen = Ab_maxlen,
            Ab_temp_coords = Ab_temp_coords,
            Ab_temp_mask = Ab_temp_mask,
            Ab_batch_mask = Ab_batch_mask,
            Ab_align_mask = Ab_align_mask,
            Ab_coords_label = Ab_coords_label,
            Ab_cdr_indexs = Ab_cdr_indexs.bool(),
            Ab_mut_range = Ab_mut_range,
            return_embeddings = return_embeddings,
            At_embeddings= At_embeddings,
            At_coords_label=At_coords_label,
            At_attentions=At_attentions,
            At_batch_mask=At_batch_mask,
            At_len=At_len,
            inter_label=inter_label
        )
    else:
        one_data = PhysicDataSamp(
            Ab_embedding = Ab_embeddings,
            Ab_phisicEmbed = Ab_phisicEmbed,
            Ab_attention = Ab_attentions,
            Ab_phisicAttention = Ab_phisicAttention,
            Ab_seqlen = Ab_seqlen,
            Ab_VHLlen = Ab_VHLlens,
            Ab_maxlen = Ab_maxlen,
            Ab_temp_coords = Ab_temp_coords,
            Ab_temp_mask = Ab_temp_mask,
            Ab_batch_mask = Ab_batch_mask,
            Ab_align_mask = Ab_align_mask,
            Ab_coords_label = Ab_coords_label,
            Ab_cdr_indexs = Ab_cdr_indexs.bool(),
            Ab_mut_range = Ab_mut_range,
            Ab_pred_coords = Ab_predCooords,
            return_embeddings = return_embeddings,

            At_embeddings= At_embeddings,
            At_coords_label=At_coords_label,
            At_attentions=At_attentions,
            At_batch_mask=At_batch_mask,
            At_len=At_len,
            inter_label=inter_label,
            Ab_abatid=Ab_abatid,
        )
    return one_data
