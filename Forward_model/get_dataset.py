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
    file_name = 'data/%s_canolize_dataset.json' %phase
    reaction_set = set()
    with open(file_name, 'r') as f:
        dataset = json.load(f)
        for product, value_ in dataset.items():
            _, cano_product = cano_smiles(product)
            materials = value_["materials"]
            cano_materials = []
            for material_ in materials:
                _, material_ = cano_smiles(material_)
                cano_materials.append(material_)
            cano_materials = sorted(cano_materials)    
            reaction = '.'.join(cano_materials)+">>"+cano_product
            if reaction not in reaction_set:
                reaction_set.add(reaction)
    return reaction_set


dataset = get_dataset(args.phase)
with open('data/%s.txt'%args.phase, 'a') as f:
    for reaction in dataset:
        f.write(reaction+"\n")
