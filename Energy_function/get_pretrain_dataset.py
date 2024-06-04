import json
import argparse
from rdkit import Chem

parser = argparse.ArgumentParser("get_dataset.py")
parser.add_argument('--phase', type=str, default='train')
args = parser.parse_args()

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

def get_dataset(phase):
    file_name = 'data/%s_canolize_dataset.json'%phase
    retro_planning_set = set()
    with open(file_name, 'r') as f:
        dataset = json.load(f)
        for product, value_ in dataset.items():
            _, cano_product = cano_smiles(product)
            materials = value_['materials']
            cano_materials = []
            for mater_ in materials:
                _, cano_mater_ = cano_smiles(mater_)
                cano_materials.append(cano_mater_)
            cano_materials = sorted(cano_materials)
            retro_planning = cano_product+">>"+'.'.join(cano_materials)
            if retro_planning not in retro_planning_set:
                retro_planning_set.add(retro_planning)
    return retro_planning_set


dataset = get_dataset(args.phase)
with open('data/%s.txt'%args.phase, 'a') as f:
    for retro_planning in dataset:
        f.write(retro_planning+"\n")

