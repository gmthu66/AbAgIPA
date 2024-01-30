import torch
from utils.general import exists
from model.interface import SingleDataSamp
from torch.nn.utils.rnn import pad_sequence
from database.inter_embed import tensorls2mat
ATOM_DIM = 3


def batch_coords(coords_ls, seqlens, maxseql, device):
    coords_mat = torch.zeros(len(seqlens), maxseql, coords_ls[0].size(-2), coords_ls[0].size(-1), device=device)
    for i_, seqlen in enumerate(seqlens):
        coords_mat[i_, :seqlen, :, :] = coords_ls[i_].squeeze(0)
    return coords_mat


def batch_Abattens(attn_ls, seqlens, maxseql, device):
    attn_mat = torch.zeros(len(seqlens), maxseql, maxseql, attn_ls[0].size(-1), device=device)
    for i_, seqlen in enumerate(seqlens):
        attn_mat[i_, :seqlen, :seqlen, :] = attn_ls[i_].squeeze(0)
    return attn_mat


def clean_input(data, device_in=None,):
    def dataToDevice(data, device_=device_in):
        if device_.type == 'cuda':
            if (data.type != device_.type) or (data.index != device_.index):
                data = data.to(device_)
        else:
            data = data.to(device_) if (data.type != device_.type) else data
        return data
    Ab_embeddings = data.Ab_embeddings
    Ab_seqlens = data.Ab_seqlens
    Ab_temp_coords = data.Ab_template_coords
    Ab_temp_mask = data.Ab_template_mask
    Ab_batch_mask = data.Ab_batch_mask
    Ab_align_mask = data.Ab_align_mask
    batch_size = len(Ab_seqlens)
    max_abseqlen = max(Ab_seqlens)

    # 初始化 Ab_batch_mask
    if not exists(Ab_batch_mask):
        Ab_batch_mask = torch.zeros(
            batch_size,
            4 * max_abseqlen,
            device=device_in,).bool()
        for i_, seq_len in enumerate(Ab_seqlens):
            Ab_batch_mask[i_, : 4 * Ab_seqlens[i_]] = torch.ones(1, 4 * Ab_seqlens[i_])
    Ab_embeddings = pad_sequence(Ab_embeddings, batch_first=True, padding_value=0.0)
    Ab_coords_label = batch_coords(data.Ab_coords_label, Ab_seqlens, max_abseqlen, device=device_in)
    Ab_attentions = batch_Abattens(data.Ab_attentions, Ab_seqlens, max_abseqlen, device=device_in)
    Ab_cdr_mask = pad_sequence(data.Ab_cdr_indexs, batch_first=True, padding_value=0.0).bool()

    # 对于没有模板输入信息的处理方式
    if not exists(Ab_temp_coords) or (Ab_temp_coords == []):
        Ab_temp_coords = torch.zeros(
            batch_size,
            4 * max_abseqlen,
            ATOM_DIM,
            device=device_in,
        ).float()  # Ab的模板坐标被初始化为原点(4种原子)
    if not exists(Ab_temp_mask) or (Ab_temp_mask == []):
        Ab_temp_mask = torch.zeros(
            batch_size,
            4 * max_abseqlen,
            device=device_in,
        ).bool()

    if not exists(Ab_align_mask):
        Ab_align_mask = torch.ones(
            batch_size,
            4 * max_abseqlen,
            device=device_in,
        ).bool()

    Ab_align_mask = Ab_align_mask & Ab_batch_mask  # Should already be masked by batch_mask anyway
    Ab_temp_coords[~Ab_temp_mask] = 0.
    for i, (tc, m) in enumerate(zip(Ab_temp_coords, Ab_temp_mask)):
        Ab_temp_coords[i][m] -= tc[m].mean(-2)

    Ab_embeddings, Ab_coords_label, Ab_attentions, Ab_temp_coords, Ab_temp_mask, Ab_batch_mask, Ab_align_mask = map(dataToDevice, [Ab_embeddings, Ab_coords_label, Ab_attentions, Ab_temp_coords, Ab_temp_mask, Ab_batch_mask, Ab_align_mask])
    data.Ab_embeddings = Ab_embeddings
    data.Ab_coords_label = Ab_coords_label
    data.Ab_attentions = Ab_attentions
    data.Ab_temp_coords = Ab_temp_coords

    data.Ab_temp_mask = Ab_temp_mask
    data.Ab_batch_mask = Ab_batch_mask
    data.Ab_align_mask = Ab_align_mask
    data.Ab_cdr_indexs = dataToDevice(Ab_cdr_mask)

    At_batch_mask = data.At_batch_mask
    if not exists(At_batch_mask):
        At_batch_mask = torch.ones(batch_size, 4 * data.At_len, device=device_in,).bool()
    data.At_embeddings = dataToDevice(torch.stack(data.At_embeddings, dim=0))
    data.At_coords_label = dataToDevice(torch.stack(data.At_coords_label, dim=0))
    data.At_attentions = dataToDevice(torch.stack(data.At_attentions, dim=0))
    data.At_batch_mask = dataToDevice(At_batch_mask)
    data.inter_label = dataToDevice(data.inter_label)

    return data, batch_size, Ab_seqlens, data.Ab_VHLlens, max_abseqlen


def onesample_data(batch_data, i_, max_abseqlen):
    one_data = SingleDataSamp(
        Ab_embedding = batch_data.Ab_embeddings[i_:i_+1, :, :],
        Ab_attention = batch_data.Ab_attentions[i_:i_+1, :, :, :],
        Ab_seqlen = batch_data.Ab_seqlens[i_],
        Ab_VHLlen = batch_data.Ab_VHLlens[i_],
        Ab_maxlen = max_abseqlen,
        Ab_temp_coords = batch_data.Ab_temp_coords[i_:i_+1, :, :],
        Ab_temp_mask = batch_data.Ab_temp_mask[i_:i_+1, :],
        Ab_batch_mask = batch_data.Ab_batch_mask[i_:i_+1, :],
        Ab_align_mask = batch_data.Ab_align_mask[i_:i_+1, :],
        Ab_coords_label = batch_data.Ab_coords_label[i_:i_+1, :, :],
        Ab_cdr_indexs = batch_data.Ab_cdr_indexs[i_:i_+1, :],
        return_embeddings = batch_data.return_embeddings,
        At_embeddings=batch_data.At_embeddings[i_:i_+1, :, :],
        At_coords_label=batch_data.At_coords_label[i_:i_+1, :, :],
        At_attentions=batch_data.At_attentions[i_:i_+1, :, :],
        At_batch_mask=batch_data.At_batch_mask[i_:i_+1, :],
        At_len=batch_data.At_len,
        inter_label=batch_data.inter_label[i_:i_+1, 1]
    )
    return one_data
