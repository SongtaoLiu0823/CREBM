import numpy as np
import pickle
from rdkit import Chem
from tqdm import tqdm
from rdkit.rdBase import DisableLog

DisableLog('rdApp.warning')

chars = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"
vocab_size = len(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

def cano_smiles(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
        if tmp is None:
            return None, smiles
        tmp = Chem.RemoveHs(tmp)
        if tmp is None:
            return None, smiles
        [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
        return tmp, Chem.MolToSmiles(tmp)
    except:
        return None, smiles


def get_chars():
    return chars

def get_vocab_size():
    return vocab_size

def get_char_to_ix():
    return char_to_ix

def get_ix_to_char():
    return ix_to_char

def get_dataset(phase):
    raw_data = []
    if phase == "train":
        with open(f"data/augmented_{phase}.txt", 'r') as f:
            for line in f.readlines():
                raw_data.append(line.strip())
    else:
        with open(f"data/{phase}.txt", 'r') as f:
            for line in f.readlines():
                raw_data.append(line.strip())
    input_list = []
    output_list = []
    for item in tqdm(raw_data):
        input_ = item.split(">>")[0]
        output_ = item.split(">>")[1]
        input_list.append(input_)
        output_list.append(output_)
    return input_list, output_list

def convert_symbols_to_inputs(input_list, output_list, max_length):
    num_samples = len(input_list)
    #input
    input_ids = np.zeros((num_samples, max_length))
    input_mask = np.zeros((num_samples, max_length))

    #output
    output_ids = np.zeros((num_samples, max_length))
    output_mask = np.zeros((num_samples, max_length))

    #for output
    label_ids = np.zeros((num_samples, max_length))

    for cnt in range(num_samples):
        input = '^' + input_list[cnt] + '$'
        output = '^' + output_list[cnt] + '$'
        
        for i, symbol in enumerate(input):
            input_ids[cnt, i] = char_to_ix[symbol]
        input_mask[cnt, :len(input)] = 1

        for i in range(len(output)-1):
            output_ids[cnt, i] = char_to_ix[output[i]]
            label_ids[cnt, i] = char_to_ix[output[i+1]]
        output_mask[cnt, :len(output)-1] = 1
    return (input_ids, input_mask, output_ids, output_mask, label_ids)
