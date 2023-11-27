import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import matplotlib.pyplot as plt
import sys
import os


def build_molecule_from_fragments(X, A, DB, odor):
    if odor == "banana":
        fragments = ["C", "O", "*c1ccco1", "*c1ccc(*)c(*)c1"]
        type_frag = 4
    elif odor == "garlic":
        fragments = ["C", "N", "S", "O", "*c1ccccc1*", "*c1ccc(*)o1", "*c1ccsc1"]
        type_frag = 7
    # number of fragments
    N_frag = X.shape[0]
    # number of features
    F = X.shape[1]
    # number of atoms
    N = 0
    # list of atoms
    atoms = []
    # indexes of attachment points
    positions = []
    # edges between atoms, (u, v, type)
    # type = 1: single
    # type = 1.5: aromatic
    # type = 2: double
    edges = []
    for v in range(N_frag):
        for i in range(type_frag):
            if X[v, i] == 1:
                if fragments[i] in ["C", "N", "S", "O"]:
                    if fragments[i] == "C":
                        atoms.append(6)
                    elif fragments[i] == "N":
                        atoms.append(7)
                    elif fragments[i] == "S":
                        atoms.append(16)
                    elif fragments[i] == "O":
                        atoms.append(8)
                    # at most 4 neighbors using the same attach point
                    positions.append([N, N, N, N])
                    N += 1
                elif fragments[i] in ["*c1ccccc1*", "*c1ccc(*)c(*)c1"]:
                    for k in range(6):
                        atoms.append(6)
                        edges.append((N + k, N + (k + 1) % 6, 1.5))
                    positions.append([N, N + 1, N + 3])
                    N += 6
                elif fragments[i] in ["*c1ccco1", "*c1ccc(*)o1"]:
                    for k in range(4):
                        atoms.append(6)
                    atoms.append(8)
                    for k in range(5):
                        edges.append((N + k, N + (k + 1) % 5, 1.5))
                    positions.append([N, N + 3])
                    N += 5
                elif fragments[i] in ["*c1ccsc1"]:
                    for k in range(4):
                        atoms.append(6)
                    atoms.append(16)
                    for k in range(5):
                        edges.append((N + k, N + (k + 1) % 5, 1.5))
                    positions.append([N])
                    N += 5

    for frag_u in range(N_frag):
        for frag_v in range(frag_u + 1, N_frag):
            if A[frag_u, frag_v] == 1:
                u = positions[frag_u][0]
                v = positions[frag_v][0]
                positions[frag_u].pop(0)
                positions[frag_v].pop(0)
                if DB[frag_u, frag_v] == 1:
                    edges.append((u, v, 2))
                else:
                    edges.append((u, v, 1))
    mol = Chem.MolFromSmiles("")
    mol = AllChem.EditableMol(mol)

    for atom in atoms:
        mol.AddAtom(Chem.Atom(atom))

    # print(atoms)
    # print(edges)

    for u, v, bond_type in edges:
        if bond_type == 1:
            mol.AddBond(u, v, Chem.BondType.SINGLE)
        elif bond_type == 1.5:
            mol.AddBond(u, v, Chem.BondType.AROMATIC)
        elif bond_type == 2:
            mol.AddBond(u, v, Chem.BondType.DOUBLE)

    return mol


odor = str(sys.argv[1])
N = int(sys.argv[2])
seed_gnn = int(sys.argv[3])
seed_grb = int(sys.argv[4])

filename = f"results/optimality/{odor}/N={N}/run_{seed_gnn}_{seed_grb}"

ans = np.load(os.path.join(filename, "ans.npy"))
X = np.load(os.path.join(filename, "X.npy"))
A = np.load(os.path.join(filename, "A.npy"))
DB = np.load(os.path.join(filename, "DB.npy"))
mol = build_molecule_from_fragments(X, A, DB, odor)

img = Draw.MolToImage(mol.GetMol())
plt.imshow(img)
plt.show()
