import argparse
from tqdm import tqdm
from rdkit import Chem

parser = argparse.ArgumentParser("augmented_dataset.py")
parser.add_argument('--phase', type=str, default='train')
args = parser.parse_args()

def get_augmented_reaction(reaction, aug_factor):
    augmented_reaction = []
    try:
        reactants, products = reaction.split(">>")
        mols_r = [Chem.MolFromSmiles(reactant) for reactant in reactants.split('.')]
        mols_p = [Chem.MolFromSmiles(product) for product in products.split('.')]
        if any(mol is None for mol in mols_r + mols_p):
            return None
        
        for mols in [mols_r, mols_p]:
            for mol in mols:
                mol = Chem.RemoveHs(mol)

        for mol in mols_r:
            for a in mol.GetAtoms():
                a.ClearProp('molAtomMapNumber')
        
        for mol in mols_p:
            for a in mol.GetAtoms():
                a.ClearProp('molAtomMapNumber')

        cano_smi_r = '.'.join(Chem.MolToSmiles(mol, isomericSmiles=True) for mol in mols_r)
        cano_smi_p = '.'.join(Chem.MolToSmiles(mol, isomericSmiles=True) for mol in mols_p)
        
        cano_reaction = cano_smi_r+">>"+cano_smi_p
        if cano_reaction not in reaction_set:
            reaction_set.add(cano_reaction)
            augmented_reaction.append(cano_smi_r+">>"+cano_smi_p)
        if aug_factor > 1:
            for _ in range(aug_factor - 1):
                smi_r = '.'.join(Chem.MolToSmiles(mol, isomericSmiles=True, doRandom=True) for mol in mols_r)
                smi_p = '.'.join(Chem.MolToSmiles(mol, isomericSmiles=True, doRandom=True) for mol in mols_p)
                reaction = smi_r+">>"+smi_p
                if reaction not in reaction_set:
                    reaction_set.add(reaction)
                    augmented_reaction.append(reaction)
    except ValueError:
        return augmented_reaction

    return augmented_reaction

reaction_list = []
with open('data/%s.txt'%args.phase, 'r') as f:
    for line in f.readlines():
        reaction_list.append(line.strip())

augmented_reaction_data = []
reaction_set = set()
for reaction in tqdm(reaction_list):
    augmented_reaction = get_augmented_reaction(reaction, 5)
    if augmented_reaction is not None:
        augmented_reaction_data += augmented_reaction


with open(f"data/augmented_{args.phase}.txt", "a") as f:
    for reaction in augmented_reaction_data:
        f.write(reaction+"\n")
