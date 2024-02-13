import warnings
import sys
import os.path as osp
import torch
import traceback
from abnumber import Chain, ChainParseError
from Bio.PDB import PDBParser, PDBIO
from Bio.SeqUtils import seq1
sys.path.append(osp.dirname(osp.dirname(__file__)))
from database.parse_utils import overlap_index
from utils.pdb import clean_pdb
from database.inter_pdb import save_residuels2pdb


def is_heavy(seq):
    chain = Chain(seq, scheme='chothia')
    return chain.is_heavy_chain()


def chain_info(aaseq):
    try:
        abnum_chain = Chain(aaseq, scheme='chothia')
        return abnum_chain
    except ChainParseError:
        return None


def rechain_pdb(pdb_file):
    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure("_", pdb_file)

    for chain in structure.get_chains():
        seq = seq1(''.join([residue.resname for residue in chain]))
        abnum_chain = Chain(seq, scheme='chothia')
        chain_id = "H" if abnum_chain.is_heavy_chain() else "L"
        try:
            chain.id = chain_id
        except ValueError:
            chain.id = chain_id + "_"
    for chain in structure.get_chains():
        if "_" in chain.id:
            chain.id = chain.id.replace("_", "")

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_file)


def renumber_pdb(
    in_pdb_file,
    out_pdb_file=None,
    scheme="chothia",
):
    """
    Renumber the pdb file.
    """
    if out_pdb_file is None:
        out_pdb_file = in_pdb_file

    clean_pdb(in_pdb_file)

    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure(
            "_",
            in_pdb_file,
        )

    chain_ls = []
    for chain in structure.get_chains():
        split_flg = True
        seq = seq1(''.join([residue.resname for residue in chain]))
        try:
            abnum_chain = Chain(seq, scheme=scheme)
        except ChainParseError:
            split_flg = split_antibody_domains(aaseq=seq)
        except:
            return False
        if split_flg is False:  continue
        numbering = abnum_chain.positions.items()
        chain_res = list(chain.get_residues())  # chian_res已经截短过tail了
        if len(abnum_chain.tail) > 0:
            chain_res = chain_res[0 : len(chain_res) - len(abnum_chain.tail)]
            print('tail in aaseq while renumber Ab/Nb: ', in_pdb_file)

        if len(chain_res) > len(numbering):
            st_i = overlap_index(seq, abnum_chain.fr1_seq)
            chain_res = chain_res[st_i:]
        assert len(chain_res) == len(numbering)

        for pdb_r, (pos, aa) in zip(chain_res, numbering):
            if aa != seq1(pdb_r.get_resname()):
                raise Exception(f"Failed to renumber PDB file {in_pdb_file}")
            pos = str(pos)[1:]
            if not pos[-1].isnumeric():
                ins = pos[-1]
                pos = int(pos[:-1])
            else:
                pos = int(pos)
                ins = ' '

            pdb_r._id = (' ', pos, ins)
        chain_ls.append(chain.id)
    if len(chain_ls) == len(list(structure.get_chains())):
        io = PDBIO()
        io.set_structure(structure)
        io.save(out_pdb_file)
    else:
        res_ls = get_resls_bychains(structure, chain_ls)
        save_residuels2pdb(res_ls, out_pdb_file)
    return True


def get_resls_bychains(structure, chain_ls):
    resls = []
    for chain in structure[0]:
        if chain.id in chain_ls:
            for res in chain:
                resls.append(res)
    return resls


def truncate_seq(seq, scheme="chothia"):
    abnum_chain = Chain(seq, scheme=scheme)
    numbering = abnum_chain.positions.items()
    seq = "".join([r[1] for r in list(numbering)])

    return seq


def split_antibody_domains(aaseq, scheme="chothia"):
    abnum_chain = None
    for i_ in range(len(aaseq), 0, -1):
        temp_seq = aaseq[:i_]
        try:
            abnum_chain = Chain(aaseq, scheme=scheme)
            return i_
        except:
            pass
    print("can't split ab seq from now aaseq")
    return False if abnum_chain is None else True


def cdrDict2Array(cdr_dict, k_='cdr'):
    cdr_index, key_ls = [], [cdri for cdri in sorted(list(cdr_dict.keys())) if k_ in cdri]
    for cdri in key_ls:
        for st, ed in cdr_dict[cdri]:
            if st != ed:
                cdr_index.extend(list(range(st, ed)))
    return cdr_index


def cdrDict2MaskTensor(cdr_dict, seqlen=None, k_='cdr'):
    # mask = torch.zeros(1, seqlen)
    mask = torch.zeros(seqlen)
    key_ls = [cdri for cdri in sorted(list(cdr_dict.keys())) if k_ in cdri]
    for cdri in key_ls:
        for st, ed in cdr_dict[cdri]:
            if st != ed:
                mask[st:ed] = 1
    return mask


def IsTwoAbSeq(aaseq, scheme="chothia"):
    split_index, total_len = 90, len(aaseq)
    for ee in range(split_index, total_len - split_index):
        info1 = chain_info(aaseq=aaseq[:ee])
        info2 = chain_info(aaseq=aaseq[ee:])
        if info1 is None or info2 is None:
            continue
        else:
            aaseq1, tail1 = abseq_and_tailseq(info1)
            aaseq2, tail2 = abseq_and_tailseq(info2)
            if len(aaseq1 + tail1) + len(aaseq2 + tail2) == len(aaseq):
                return ee, info1, info2
    return None, None, None


def abseq_and_tailseq(chain_info):
    def joinseq(seqls):
        totalseq = ""
        for seqi in seqls:
            if isinstance(seqi, str):  totalseq += seqi
        return totalseq

    aaseq_list = [getattr(chain_info, kname) for kname in dir(chain_info) if '_seq' in kname]
    aaseq = joinseq(aaseq_list)
    tail_seq = chain_info.tail
    return aaseq, tail_seq
