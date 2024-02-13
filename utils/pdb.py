import time
import sys
import os.path as osp
from typing import Union, List
import requests
import warnings
from os.path import splitext, basename
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO, Residue
from Bio.SeqUtils import seq1
from bisect import bisect_left, bisect_right
import torch
import numpy as np

sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils.constants import ResnDICT
from utils.coordinates import place_fourth_atom
from utils.fasta import get_fasta_chain_seq
from utils.general import _aa_1_3_dict, exists
chain_dict = {"A": "H", "B": "L", "H": "H", "L": "L"}


def renumber_pdb(old_pdb, renum_pdb=None):
    if not exists(renum_pdb):
        renum_pdb = old_pdb

    success = False
    time.sleep(5)
    for i in range(10):
        try:
            with open(old_pdb, 'rb') as f:
                response = requests.post(
                    'http://www.bioinf.org.uk/abs/abnum/abnumpdb.cgi',
                    params={
                        "plain": "1",
                        "output": "-HL",
                        "scheme": "-c"
                    },
                    files={"pdb": f},
                )

            success = response.status_code == 200 and not ("<html>" in response.text)

            if success:
                break
            else:
                time.sleep((i + 1) * 5)
        except requests.exceptions.ConnectionError:
            time.sleep(60)

    # if success:
    if success:
        new_pdb_data = response.text
        with open(renum_pdb, "w") as f:
            f.write(new_pdb_data)
    else:
        print(
            "Failed to renumber PDB. This is likely due to a connection error or a timeout with the AbNum server."
        )


def count_pdb_chains(pdb_file):
    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure("_", pdb_file)

    l = len(list(structure.get_chains()))

    return l


def reorder_pdb_chains(pdb_file, chain_order):
    """Reorder the chains in a PDB file and update residue numbers"""

    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure("_", pdb_file)

    chains = list(structure.get_chains())
    if len(chains) != len(chain_order):
        raise ValueError(
            f"Number of chains in PDB file ({len(chains)}) does not match number of chains in chain order ({len(chain_order)})"
        )

    chain_order = [c.upper() for c in chain_order]
    sorted_chains = sorted(chains, key=lambda c: chain_order.index(c.id))

    new_structure = PDB.Structure.Structure("_")
    new_model = PDB.Model.Model(0)
    new_structure.add(new_model)
    atom_num = 1
    for chain in sorted_chains:
        new_chain = PDB.Chain.Chain(chain.id)
        new_model.add(new_chain)
        for residue in chain.get_residues():
            new_residue = PDB.Residue.Residue(
                residue.id,
                residue.resname,
                residue.segid,
            )
            new_chain.add(new_residue)
            for atom in residue:
                new_atom = PDB.Atom.Atom(
                    atom.name,
                    atom.coord,
                    atom.occupancy,
                    atom.bfactor,
                    atom.altloc,
                    atom.fullname,
                    atom_num,
                    atom.element,
                )
                new_residue.add(new_atom)
                atom_num += 1

    io = PDBIO()
    io.set_structure(new_structure)
    io.save(pdb_file)


def get_atom_coord(residue, atom_type):
    if exists(residue) and atom_type in residue:
        return residue[atom_type].get_coord()
    else:
        return [0, 0, 0]


def get_cb_or_ca_coord(residue):
    if not exists(residue):
        return [0, 0, 0]

    if 'CB' in residue:
        return residue['CB'].get_coord()
    elif 'CA' in residue:
        return residue['CA'].get_coord()
    else:
        return [0, 0, 0]


def get_continuous_ranges(residues):
    """ Returns ranges of residues which are continuously connected (peptide bond length 1.2-1.45 Å) """
    dists = []
    for res_i in range(len(residues) - 1):
        dists.append(
            np.linalg.norm(
                np.array(get_atom_coord(residues[res_i], "C")) -
                np.array(get_atom_coord(residues[res_i + 1], "N"))))

    ranges = []
    start_i = 0
    for d_i, d in enumerate(dists):
        if d > 1.45 or d < 1.2:
            ranges.append((start_i, d_i + 1))
            start_i = d_i + 1
        if d_i == len(dists) - 1:
            ranges.append((start_i, None))

    return ranges


def place_missing_cb_o(atom_coords):
    cb_coords = place_fourth_atom(
        atom_coords['C'],
        atom_coords['N'],
        atom_coords['CA'],
        torch.tensor(1.522),
        torch.tensor(1.927),
        torch.tensor(-2.143),
    )
    o_coords = place_fourth_atom(
        torch.roll(atom_coords['N'], shifts=-1, dims=0),
        atom_coords['CA'],
        atom_coords['C'],
        torch.tensor(1.231),
        torch.tensor(2.108),
        torch.tensor(-3.142),
    )

    bb_mask = get_atom_coords_mask(atom_coords['N']) & get_atom_coords_mask(
        atom_coords['CA']) & get_atom_coords_mask(atom_coords['C'])
    missing_cb = (get_atom_coords_mask(atom_coords['CB']) & bb_mask) == 0
    if missing_cb.sum() == cb_coords.size(0):
        atom_coords['CB'] = cb_coords
        print('place_missing_cb_o: all residues CB atom missing')
    else:
        atom_coords['CB'][missing_cb] = cb_coords[missing_cb]

    bb_mask = get_atom_coords_mask(
        torch.roll(
            atom_coords['N'],
            shifts=-1,
            dims=0,
        )) & get_atom_coords_mask(atom_coords['CA']) & get_atom_coords_mask(
            atom_coords['C'])
    missing_o = (get_atom_coords_mask(atom_coords['O']) & bb_mask) == 0
    atom_coords['O'][missing_o] = o_coords[missing_o]


def pdbresls2fasta(residues, fasta_file):
    now_ch, context, aaseq = None, [], []
    if 'residue' in dir(residues[0]):  residues = [res.residue for res in residues]
    for resi in residues:
        if now_ch is None:
            now_ch = resi.full_id[2]
            aaseq.append(seq1(resi.get_resname()))
            context.append(f'>{now_ch}\n')
        elif now_ch != resi.full_id[2]:
            context.append('{}\n'.format(''.join(aaseq)))
            aaseq = []
            aaseq.append(seq1(resi.get_resname()))
            now_ch = resi.full_id[2]
            context.append(f'>{now_ch}\n')
        else:
            # aaseq.append(ResnDICT[resi.resname])
            aaseq.append(seq1(resi.get_resname()))
    context.append('{}\n'.format(''.join(aaseq)))
    with open(fasta_file, 'w') as w:
        for l_ in context:
            w.write(l_)


def reorder_ch(s_chains):
    temp_ls = [None] * 2
    for i_, ch in enumerate(s_chains):
        if (chain_dict[ch.id] == 'L' and i_ == 0) or (chain_dict[ch.id] == 'H' and i_ == 1):
            temp_ls[1 - i_] = ch
        else:
            temp_ls[i_] = ch
    if None in temp_ls: temp_ls.remove(None)
    return temp_ls


def get_atom_coords(pdb_file, fasta_file=None):
    p = PDBParser()
    file_name, r_rangels = splitext(basename(pdb_file))[0], []
    structure = p.get_structure(
        file_name,
        pdb_file,
    )
    residues, dict_range = [], {}  # dict_range中是 H chain和 L chain中氨基酸顺序序号
    fasta_file_flag = check_chs(structure, fasta_file)  # 查看fasta file中是否有多个chains, 防止fasta file错误
    if fasta_file_flag:
        for ch_i, chain in enumerate(reorder_ch(structure.get_chains())):
            pdb_seq = get_pdb_chain_seq(
                pdb_file,
                chain.id,
            )

            fasta_seq, ch_ls = get_fasta_chain_seq(fasta_file, chain_dict[chain.id])  # 有的时候fasta中的aaseq与pdb的aaseq没有对齐
            if fasta_seq is None: continue

            chain_residues = list(chain.get_residues())  # 这里有时会与fasta_seq并不能一一对应
            continuous_ranges = get_continuous_ranges(chain_residues)

            fasta_residues = [None for _ in range(len(fasta_seq))]
            fasta_r, dict_range[ch_i] = (0, 0), []
            for pdb_r in continuous_ranges:
                fasta_r_start = fasta_seq[fasta_r[1]:].index(pdb_seq[pdb_r[0]:pdb_r[1]]) + fasta_r[1]
                fasta_r_end = (len(pdb_seq) if pdb_r[1] == None else pdb_r[1]) - pdb_r[0] + fasta_r_start
                fasta_r = (fasta_r_start, fasta_r_end)
                fasta_residues[fasta_r[0]:fasta_r[1]] = chain_residues[pdb_r[0]:pdb_r[1]]
                dict_range[ch_i].append(fasta_r)
            residues += fasta_residues  # 获得空间上连续的几段多肽索引序列

        r_rangels.append(dict_range)
    else:
        residues = list(structure.get_residues())
        # pdbresls2fasta(residues, fasta_file)
    if len(residues) < 10:
        return None, None
    r_rangels = ([False] + r_rangels) if None not in residues else ([True] + r_rangels)
    n_coords = torch.tensor([get_atom_coord(r, 'N') for r in residues])  # 对于没有的原子, 会返回0
    ca_coords = torch.tensor([get_atom_coord(r, 'CA') for r in residues])
    c_coords = torch.tensor([get_atom_coord(r, 'C') for r in residues])
    cb_coords = torch.tensor([get_atom_coord(r, 'CB') for r in residues])
    cb_ca_coords = torch.tensor([get_cb_or_ca_coord(r) for r in residues])
    o_coords = torch.tensor([get_atom_coord(r, 'O') for r in residues])

    atom_coords = {}
    atom_coords['N'] = n_coords
    atom_coords['CA'] = ca_coords
    atom_coords['C'] = c_coords
    atom_coords['CB'] = cb_coords
    atom_coords['CBCA'] = cb_ca_coords
    atom_coords['O'] = o_coords
    #  (N, Cα, C, and Cβ)
    place_missing_cb_o(atom_coords)
    # check_nanInTensor(atom_coords)
    return atom_coords, r_rangels


def get_atom_coords_mask(coords):
    mask = torch.ByteTensor([1 if sum(_) != 0 else 0 for _ in coords])
    mask = mask & (1 - torch.any(torch.isnan(coords), dim=1).byte())
    return mask


def get_atom_coords_mask_for_dict(atom_coords):
    atom_coords_masks = {}
    for atom, coords in atom_coords.items():
        atom_coords_masks[atom] = get_atom_coords_mask(coords)

    return atom_coords_masks


def check_nanInTensor(t_dict):
    for k_ in t_dict.keys():
        if torch.isnan(t_dict[k_]).any():
            print('\nnan in tensor:', k_)


def check_chs(structure, fasta_f):
    chain_dict = {"A": "H", "B": "L", "H": "H", "L": "L"}
    pdb_chls = sorted([chain_dict[ch.id] for ch in structure.get_chains()])
    if isinstance(fasta_f, dict):
        fasta_chls = list(fasta_f.keys())
    else:
        _, fasta_chls = get_fasta_chain_seq(fasta_f, 'H', check_ch=True)
    if len(fasta_chls) < 1:
        residues = list(structure.get_residues())
        pdbresls2fasta(residues, fasta_f)
        _, fasta_chls = get_fasta_chain_seq(fasta_f, 'H', check_ch=True)
    if len(pdb_chls) < len(fasta_chls):
        trunc_fastaf(fasta_f, pdb_chls)
        print('pdb chain list not match with fasta file chain list, fixed')
    fasta_file = False if len(fasta_chls) < 1 else True
    return fasta_file


def trunc_fastaf(fasta_f, ch_ls):
    with open(fasta_f, 'r') as r:
        context = r.readlines()
    i_ = 0
    with open(fasta_f, 'w') as w:
        while i_ < len(context):
            if lsAnyInStr(ch_ls, context[i_]):
                w.write(context[i_])
                w.write(context[i_ + 1])
            i_ += 2


def lsAnyInStr(ls, s_):
    for i_ in ls:
        if i_ in s_:
            return True
    return False


def pdb2fasta(pdb_file, num_chains=None):
    """Converts a PDB file to a fasta formatted string using its ATOM data"""
    pdb_id = basename(pdb_file).split('.')[0]
    parser = PDBParser()
    structure = parser.get_structure(
        pdb_id,
        pdb_file,
    )

    real_num_chains = len([0 for _ in structure.get_chains()])
    if num_chains is not None and num_chains != real_num_chains:
        print('WARNING: Skipping {}. Expected {} chains, got {}'.format(
            pdb_file, num_chains, real_num_chains))
        return ''

    fasta = ''
    for chain in reorder_ch(structure.get_chains()):
        id_ = chain.id
        seq = seq1(''.join([residue.resname for residue in chain]))
        fasta += '>{}:{}\t{}\n'.format(pdb_id, id_, len(seq))
        max_line_length = 80
        for i in range(0, len(seq), max_line_length):
            fasta += f'{seq[i:i + max_line_length]}\n'
    return fasta


def get_pdb_chain_seq(
    pdb_file,
    chain_id,
):
    p = PDBParser()
    file_name = splitext(basename(pdb_file))[0]
    structure = p.get_structure(
        file_name,
        pdb_file,
    )

    pdb_seq = None
    for chain in structure.get_chains():
        if chain.id == chain_id:
            pdb_seq = "".join(
                [seq1(r.get_resname()) for r in chain.get_residues()])

    return pdb_seq


def cdr_indices(
    chothia_pdb_file,
    cdr,
    offset_heavy=True,
):
    """Gets the index of a given CDR loop"""
    cdr_chothia_range_dict = {
        "h1": (26, 32),
        "h2": (52, 56),
        "h3": (95, 102),
        "l1": (24, 34),
        "l2": (50, 56),
        "l3": (89, 97)
    }

    cdr = str.lower(cdr)
    assert cdr in cdr_chothia_range_dict.keys()

    chothia_range = cdr_chothia_range_dict[cdr]
    chain_id = cdr[0].upper()

    parser = PDBParser()
    pdb_id = basename(chothia_pdb_file).split('.')[0]
    structure = parser.get_structure(
        pdb_id,
        chothia_pdb_file,
    )
    cdr_chain_structure = None
    for chain in structure.get_chains():
        if chain.id == chain_id:
            cdr_chain_structure = chain
            break
    if cdr_chain_structure is None:
        print("PDB must have a chain with chain id \"[PBD ID]:{}\"".format(
            chain_id))
        sys.exit(-1)

    residue_id_nums = [res.get_id()[1] for res in cdr_chain_structure]

    # Binary search to find the start and end of the CDR loop
    cdr_start = bisect_left(
        residue_id_nums,
        chothia_range[0],
    )
    cdr_end = bisect_right(
        residue_id_nums,
        chothia_range[1],
    ) - 1

    if len(get_pdb_chain_seq(
            chothia_pdb_file,
            chain_id=chain_id,
    )) != len(residue_id_nums):
        print('ERROR in PDB file ' + chothia_pdb_file)
        print('residue id len', len(residue_id_nums))

    if chain_id == "L" and offset_heavy:
        heavy_seq_len = get_pdb_chain_seq(
            chothia_pdb_file,
            chain_id="H",
        )
        cdr_start += len(heavy_seq_len)
        cdr_end += len(heavy_seq_len)

    return cdr_start, cdr_end


def get_cdr_range_dict(
    chothia_pdb_file,
    heavy_only=False,
    light_only=False,
    offset_heavy=True,
):
    cdr_names = ["h1", "h2", "h3", "l1", "l2", "l3"]
    if heavy_only:
        cdr_names = cdr_names[:3]
    if light_only:
        cdr_names = cdr_names[3:]

    cdr_range_dict = {
        cdr: cdr_indices(
            chothia_pdb_file,
            cdr,
            offset_heavy=offset_heavy,
        )
        for cdr in cdr_names
    }

    return cdr_range_dict


def h3_indices(chothia_pdb_file):
    """Gets the index of the CDR H3 loop"""

    return cdr_indices(chothia_pdb_file, cdr="h3")


def get_chain_numbering(
    pdb_file,
    chain_id,
):
    seq = []
    parser = PDBParser()
    structure = parser.get_structure("_", pdb_file)
    for chain in structure.get_chains():
        if chain.id == chain_id:
            for r in chain.get_residues():
                res_num = str(r._id[1]) + r._id[2]
                res_num = res_num.replace(" ", "")
                seq.append(res_num)

            return seq


def save_PDB(
    out_pdb: str,
    coords: torch.Tensor,
    seq: str,
    chains: List[str] = None,
    error: torch.Tensor = None,
    delim: Union[int, List[int]] = None,
    atoms=['N', 'CA', 'C', 'O', 'CB'],
    write_pdb=True,
) -> None:
    """
    Write set of N, CA, C, O, CB coords to PDB file
    """

    if not exists(chains):
        chains = ["H", "L"]

    if type(delim) == type(None):
        delim = -1
    elif type(delim) == int:
        delim = [delim]

    if not exists(error):
        error = torch.zeros(len(seq))

    pdb_string = ""
    k = 0
    for r, residue in enumerate(coords):
        AA = _aa_1_3_dict[seq[r]]
        for a, atom in enumerate(residue):
            if AA == "GLY" and atoms[a] == "CB": continue
            x, y, z = atom
            chain_id = chains[np.where(np.array(delim) - r > 0)[0][0]]
            pdb_string += "ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n" % (
                k + 1, atoms[a], AA, chain_id, r + 1, x, y, z, 1, error[r])
            k += 1

            if k in delim:
                pdb_string += "TER  %5d      %3s %s%4d\n" % (
                    k + 1, AA, chain_id, r + 1)
                k += 1
                
    pdb_string += "END\n"

    if write_pdb:
        with open(out_pdb, "w") as f:
            f.write(pdb_string)

    return pdb_string


def write_pdb_bfactor(
    in_pdb_file,
    out_pdb_file,
    bfactor,
    b_chain=None,
):
    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure(
            "_",
            in_pdb_file,
        )

    i = 0
    for chain in structure.get_chains():
        if exists(b_chain) and chain._id != b_chain:
            continue

        for r in chain.get_residues():
            [a.set_bfactor(bfactor[i]) for a in r.get_atoms()]
            i += 1

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb_file)


def clean_pdb(pdb_file):
    with open(pdb_file, "r") as f:
        lines = f.readlines()

    with open(pdb_file, "w") as f:
        for l in lines:
            if "ATOM" in l:
                f.write(l)
