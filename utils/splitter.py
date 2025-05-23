import random
from collections import defaultdict
from typing import List, Set, Union, Dict
import torch
import numpy as np

from tqdm import tqdm
from rdkit import Chem
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain assert from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


def scaffold_split(dataset, smiles_list, task_idx=None, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                   return_smiles=False):
    """
      Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
      Split dataset by Bemis-Murcko scaffolds
      This function can also ignore examples containing null values for a
      selected task when splitting. Deterministic split
      :param dataset: pytorch geometric dataset obj
      :param smiles_list: list of smiles corresponding to the dataset obj
      :param task_idx: column idx of the data.y tensor. Will filter out
      examples with null value in specified task column of the data.y tensor
      prior to splitting. If None, then no filtering
      :param null_value: float that specifies null value in data.y to filter if
      task_idx is provided
      :param frac_train:
      :param frac_valid:
      :param frac_test:
      :param return_smiles:
      :return: train, valid, test slices of the input dataset obj. If
      return_smiles = True, also returns ([train_smiles_list],
      [valid_smiles_list], [test_smiles_list])
    """

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    if task_idx is not None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to not null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, s in smiles_list:
        scaffold = generate_scaffold(s, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)

    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    if not return_smiles:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i][1] for i in train_idx]
        valid_smiles = [smiles_list[i][1] for i in valid_idx]
        test_smiles = [smiles_list[i][1] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles)


def split_train_val_test_idx_split_file(split_path, sort=False):
    npz_data = np.load(split_path, allow_pickle=True)
    train_idx, valid_idx, test_idx = npz_data["idx_train"], npz_data["idx_val"], npz_data["idx_test"]

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0

    if sort:
        train_idx = sorted(train_idx)
        valid_idx = sorted(valid_idx)
        test_idx = sorted(test_idx)

    return train_idx, valid_idx, test_idx


def scaffold_split_train_val_test(index, smiles_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1, sort=False):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    index = np.array(index)

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_index, val_index, test_index = index[train_idx], index[valid_idx], index[test_idx]

    if sort:
        train_index = sorted(train_index)
        val_index = sorted(val_index)
        test_index = sorted(test_index)

    return train_index, val_index, test_index


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.

    :param mols: A list of smiles strings or RDKit molecules.
    :param use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
    to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        if Chem.MolFromSmiles(mol) != None:
            scaffold = generate_scaffold(mol)
            if use_indices:
                scaffolds[scaffold].add(i)
            else:
                scaffolds[scaffold].add(mol)

    return scaffolds


