import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)


def get_mol_fragments(mol, add_dummy=True, remove_label=True):
    # generate aromatic fragments and unique aliphatic atoms from a rdkit Molecule.
    # mol: fragmented molecule
    # add_dummy: keep subsitution sites as dummy atom (label: *). Subsitution sites on aliphatic atoms are always removed.
    # remove_label: remove number of subsitution site label.

    split_bonds = set()
    split_pattern_list = [
        Chem.MolFromSmarts("*-*"),  # All single bonds
        Chem.MolFromSmarts("A=A"),  # All aliphatic double bonds
        Chem.MolFromSmarts("A#A"),  # All aliphatic triple bonds
    ]
    # determine unique bond indices
    for split_pattern in split_pattern_list:
        split_bonds.update(
            {
                mol.GetBondBetweenAtoms(a1, a2).GetIdx()
                for a1, a2 in mol.GetSubstructMatches(split_pattern)
            }
        )

    # if no bonds are selected.
    if not split_bonds:
        return [Chem.MolToSmiles(mol)]

    # fragment on bonds and extract fragments
    fragmented_mol = Chem.FragmentOnBonds(mol, split_bonds, addDummies=add_dummy)

    fragment_list = list(Chem.GetMolFrags(fragmented_mol, asMols=True))
    # remove number from subsitution site label, if requested
    if remove_label:
        for fragment in fragment_list:
            for atom in fragment.GetAtoms():
                atom.SetIsotope(0)

    # for aliphatic atoms with substitution sites: remove substitution site
    subsitution_site_aliphatic = Chem.MolFromSmarts("[$([#0]~[A])]")
    final_fragment_list = []
    for fragment in fragment_list:
        fragment = AllChem.DeleteSubstructs(fragment, subsitution_site_aliphatic)
        try:
            Chem.SanitizeMol(fragment)
            fragment = Chem.RemoveHs(fragment)
        except Exception as err:
            display(fragment)
            raise err
        final_fragment_list.append(fragment)

    return [Chem.MolToSmiles(fragment) for fragment in final_fragment_list]


odor_list = [
    "alliaceous",
    "almond",
    "amber",
    "animal",
    "apple",
    "apricot",
    "balsamic",
    "banana",
    "berry",
    "camphoraceous",
    "cherry",
    "cinnamyl",
    "citrus",
    "coconut",
    "coffee",
    "garlic",
    "grape",
    "jasmine",
    "lemon",
    "lily",
    "melon",
    "peach",
    "pear",
    "pine",
    "pineapple",
    "raspberry",
    "vanilla",
]

aroma_df = pd.read_csv("data/SMILES_odor_mapping.tsv", sep="\t").query(
    "odor.isin(@odor_list)"
)

# removing stereochemistry
aroma_df["nonstereo_smiles"] = aroma_df["SMILES"].apply(
    lambda smi: Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
)

# creating matrix with comounds x odors
unique_odors = aroma_df.odor.unique()
full_odor_df = pd.DataFrame(
    index=pd.Index(aroma_df.nonstereo_smiles.unique(), name="nonstereo_smiles"),
    columns=pd.Index(unique_odors, name="odor"),
)

# setting all values to False
full_odor_df[:] = False
# changing dtype to bool
full_odor_df = full_odor_df.astype(bool)

# setting all values to True where odor is recorded
for index, row in aroma_df.iterrows():
    full_odor_df.loc[row["nonstereo_smiles"], row["odor"]] = True
full_odor_df.reset_index(inplace=True)

full_odor_df["fragment_list"] = full_odor_df.nonstereo_smiles.apply(
    lambda smi: get_mol_fragments(Chem.MolFromSmiles(smi), add_dummy=True)
)
full_odor_df["fragment_set"] = full_odor_df["fragment_list"].apply(set)
full_odor_df["fragment_count"] = full_odor_df["fragment_list"].apply(len)

import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import os

# types of fragments with banana as the target odor
frag_types = [
    "C",
    "N",
    "S",
    "O",
    "*c1ccccc1",
    "*c1ccccc1*",
    "*c1ccco1",
    "*c1ccc(*)o1",
    "c1ccsc1",
    "*c1ccsc1",
]
# covalence for each type of fragment
covalences = [4, 3, 2, 2, 2, 2, 1]


def get_frag_type(frag_smiles):
    for index, smiles in enumerate(frag_types):
        # after ignoring the attachment positions, there are seven types of fragments
        if frag_smiles == smiles:
            if index < 4:
                return index
            elif index in [4, 5]:
                return 4
            elif index in [6, 7]:
                return 5
            elif index in [8, 9]:
                return 6


def build_graph_for_fragments(smiles, mol_index, odor, data_index):
    # build graph structure for fragment-based molecules
    # smiles: SMILES of fragmented molecule
    # mol_index: index of fragmented molecule in original dataset
    # odor: if this molecule has target odor
    # data_index: the new index of this molecule in preprocessed dataset

    mol = Chem.MolFromSmiles(smiles)
    split_bonds = set()
    split_pattern_list = [
        Chem.MolFromSmarts("*-*"),  # All single bonds
        Chem.MolFromSmarts("A=A"),  # All aliphatic double bonds
        Chem.MolFromSmarts("A#A"),  # All aliphatic triple bonds
    ]

    # determine unique bond indices
    for split_pattern in split_pattern_list:
        split_bonds.update(
            {
                mol.GetBondBetweenAtoms(a1, a2).GetIdx()
                for a1, a2 in mol.GetSubstructMatches(split_pattern)
            }
        )

    # if no bonds are selected.
    if not split_bonds:
        return 0

    # fragment on bonds and extract fragments
    frag_mol = Chem.FragmentOnBonds(mol, split_bonds, addDummies=False)

    # number of atoms
    n_atoms = len(mol.GetAtoms())
    # this list provides the mapping between atoms and fragments
    frag_list = list(Chem.GetMolFrags(frag_mol, asMols=False))
    # list of smiles of each fragment
    frag_smiles_list = get_mol_fragments(mol)
    # number of fragments
    N = len(frag_list)
    # number of features
    F = len(covalences) + 4 + 5 + 1
    # edge set
    edges = []
    # double bond tag for each edge
    db_tag = []
    # feature matrix
    X = np.zeros((N, F), dtype=np.int8)
    # number of neighbors for each fragment
    neighbors = np.zeros(N, dtype=np.int8)
    # number of hydrogen atoms for each fragment
    counts = np.zeros(N, dtype=np.int8)
    # index of fragment that each atom belongs to
    atom_index = np.zeros(n_atoms, dtype=np.int8)

    for frag_index in range(N):
        # get the fragment type
        frag_type = get_frag_type(frag_smiles_list[frag_index])
        # update corresponding feature
        X[frag_index, frag_type] = 1
        # get the maximal number of hydrogen atoms associated with this fragment
        counts[frag_index] = covalences[frag_type]
        # index each atom in this fragment
        for atom in frag_list[frag_index]:
            atom_index[atom] = frag_index

    # build graph for fragments
    for atom_u in mol.GetAtoms():
        for atom_v in atom_u.GetNeighbors():
            u = atom_u.GetIdx()
            v = atom_v.GetIdx()
            frag_u = atom_index[u]
            frag_v = atom_index[v]
            # record each edge between two different fragments
            if frag_u != frag_v:
                # update the number of neighbors of fragment frag_u
                neighbors[frag_u] += 1
                # update edge set
                edges.append([frag_u, frag_v])
                # get the bond type
                db_tag.append(0)
                bond_type = mol.GetBondBetweenAtoms(u, v).GetBondTypeAsDouble()
                if bond_type == 3.0 and not odor:
                    return 0
                # for both odors considered, there is no triple bond
                assert bond_type in [1.0, 2.0]
                # update the number of hydrogen atoms associated with fragment frag_u
                counts[frag_u] -= int(bond_type)
                # update double bond feature and double bond tag
                if bond_type == 2.0:
                    X[frag_u, -1] = 1
                    db_tag[-1] = 1

    # update features for number of neighbors (1,2,3,4) and hydrogen atoms (0,1,2,3,4)
    for frag_index in range(N):
        X[frag_index, len(covalences) + neighbors[frag_index] - 1] = 1
        X[frag_index, len(covalences) + 4 + counts[frag_index]] = 1

    edges = np.transpose(np.array(edges))
    db_tag = np.array(db_tag)

    # add some constraints to filter negative data for banana odor
    if not odor:
        M = edges.shape[1] // 2
        if (
            N > 10
            or sum(X[:, 2]) + sum(X[:, 4]) + sum(X[:, 5]) + sum(X[:, 6]) == 0
            or M - (N - 1) > 1
        ):
            return 0
        # bounds for nitrogen atoms
        if sum(X[:, 1]) > N // 4:
            return 0
        # bounds for sulfur atoms
        if sum(X[:, 2]) > N // 4:
            return 0
        # bounds for oxygen atoms
        if sum(X[:, 3]) > N // 4:
            return 0
        # bounds for aromatic ring
        if sum(X[:, 4]) + sum(X[:, 5]) + sum(X[:, 6]) > 2:
            return 0
        # bounds for double bonds
        if sum(db_tag) // 2 > N // 4:
            return 0

    # print(edges)
    # print(db_tag)
    # print(X)

    # construct a data corresponding to this molecule for training
    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=torch.tensor(edges, dtype=torch.long),
        y=torch.tensor([odor], dtype=torch.long),
        db_tag=torch.tensor(db_tag, dtype=torch.long),
        index=torch.tensor(mol_index, dtype=torch.long),
        smiles=smiles,
    )
    dir = "data/garlic/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(data, os.path.join(dir, f"data_{data_index}.pt"))
    return 1


odor = "garlic"
data_index = 0
# remove improper molecules, including molecules containing sulfur atoms with covalence 6
removed_list = [35, 1580, 1772]

# process all molecules with target odor
single_odor_df = full_odor_df.query(odor)[["nonstereo_smiles", "fragment_set"]]
for row in single_odor_df.itertuples():
    smiles = getattr(row, "nonstereo_smiles")
    index = getattr(row, "Index")
    if index in removed_list:
        continue
    res = build_graph_for_fragments(smiles, index, 1, data_index)
    data_index += res

# number of molecules with target odor
print("number of molecules with target odor:", data_index)

# get the list of fragments from molecules with target odor
unique_fragments = set.union(*single_odor_df.fragment_set.tolist())
unique_compounds = single_odor_df.nonstereo_smiles.unique()

# molecules without target odor but consist of fragments in unique_compounds
non_odor_cpds = full_odor_df.query("~nonstereo_smiles.isin(@unique_compounds)")
non_odor_cpds = non_odor_cpds.loc[
    non_odor_cpds["fragment_set"].apply(lambda row: row.issubset(unique_fragments))
]

# process all molecules without target odor but consist of fragments in unique_compounds
for row in non_odor_cpds.itertuples():
    smiles = getattr(row, "nonstereo_smiles")
    index = getattr(row, "Index")
    if index in removed_list:
        continue
    res = build_graph_for_fragments(smiles, index, 0, data_index)
    data_index += res

# number of molecules for training
print("number of molecules for training:", data_index)
