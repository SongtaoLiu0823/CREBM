import numpy as np
import torch
import torch.nn as nn
import json
import os
import pandas as pd
import math
import scipy
import argparse
import rdkit.Chem.AllChem as AllChem

from copy import deepcopy
from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun
from pathlib import Path
from typing import Dict, List
from scipy import sparse
from tqdm import trange
from rdkit import Chem, DataStructs
from model import TemplateNN_Highway
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from forward_model import ForwardTransformerConfig, ForwardTransformer, get_input_mask, get_output_mask, get_mutual_mask

def mol_smi_to_count_fp(mol_smi: str, radius: int = 2, fp_size: int = 32681, dtype: str = "int32") -> scipy.sparse.csr_matrix:
    fp_gen = GetMorganGenerator(radius=radius, useCountSimulation=True, includeChirality=True, fpSize=fp_size)
    mol = Chem.MolFromSmiles(mol_smi)
    uint_count_fp = fp_gen.GetCountFingerprint(mol)
    count_fp = np.empty((1, fp_size), dtype=dtype)
    DataStructs.ConvertToNumpyArray(uint_count_fp, count_fp)
    return sparse.csr_matrix(count_fp, dtype=dtype)


class Proposer:
    def __init__(self, infer_config: Dict) -> None:
        super().__init__()
        self.device = device_neuralsym

        print(f"Loading templates from file: {infer_config['templates_file']}")
        with open(f"{DATA_FOLDER}/{infer_config['templates_file']}", 'r') as f:
            templates = f.readlines()
        self.templates_filtered = []
        for p in templates:
            pa, cnt = p.strip().split(': ')
            if int(cnt) >= infer_config['min_freq']:
                self.templates_filtered.append(pa)
        print(f'Total number of template patterns: {len(self.templates_filtered)}')

        self.model, self.indices = self.build_model(infer_config)
        self.model.eval()
        print('Done initializing proposer\n')

    def build_model(self, infer_config: Dict):
         # load model from checkpoint
        checkpoint = torch.load(
            f"{CHECKPOINT_FOLDER}/{infer_config['expt_name']}.pth.tar",
            map_location=self.device,
        )
        model = TemplateNN_Highway(
            output_size=len(self.templates_filtered),
            size=infer_config['hidden_size'],
            num_layers_body=infer_config['depth'],
            input_size=infer_config['final_fp_size']
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)

        indices = np.loadtxt(f"{DATA_FOLDER}/variance_indices.txt").astype('int')
        return model, indices

    def propose(self, 
                smi: str,
                topk: int = 5,
                **kwargs) -> List[Dict[str, List]]:

        answer = []
        with torch.no_grad():
            prod_fp = mol_smi_to_count_fp(smi, infer_config['radius'], infer_config['orig_fp_size'])
            logged = sparse.csr_matrix(np.log(prod_fp.toarray() + 1))
            final_fp = logged[:, self.indices]
            final_fp = torch.as_tensor(final_fp.toarray()).float().to(self.device)

            outputs = self.model(final_fp)
            outputs = nn.Softmax(dim=1)(outputs)
            preds = torch.topk(outputs, k=100, dim=1)[1].squeeze(dim=0).cpu().numpy()

            aim_size = topk
            for idx in preds:
                score = outputs[0, idx.item()].item()
                template = self.templates_filtered[idx.item()]
                try:
                    rxn = rdchiralReaction(template)
                    prod = rdchiralReactants(smi)
                    precs = rdchiralRun(rxn, prod)
                except:
                    precs = 'N/A'
                if precs != 'N/A' and precs != []:
                    reactants = set(precs[0].split("."))
                    num_valid_reactant = 0
                    sms = set()
                    for r in reactants:
                        m = Chem.MolFromSmiles(r)
                        if m is not None:
                            num_valid_reactant += 1
                            sms.add(Chem.MolToSmiles(m))
                    if num_valid_reactant != len(reactants):
                        continue
                    if len(sms):
                        answer.append([sorted(list(sms)), -math.log10(score+1e-10)]) # Tuple[precs, score] where precs is a List[str]
                        aim_size -= 1
                if aim_size == 0:
                    break
        return answer[:topk]


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


def load_dataset(split):
    file_name = "%s_dataset.json" % split
    file_name = os.path.expanduser(file_name)
    dataset = [] # (product_smiles, materials_smiles, depth)
    with open(file_name, 'r') as f:
        _dataset = json.load(f)
        for _, reaction_trees in _dataset.items():
            product = reaction_trees['1']['retro_routes'][0][0].split('>')[0]
            product = Chem.MolToSmiles(Chem.MolFromSmiles(product))
            _, product = cano_smiles(product)
            max_num_materials = 0
            final_material_list = None
            for i in range(1, int(reaction_trees['num_reaction_trees'])+1):
                if len(reaction_trees[str(i)]['materials']) > max_num_materials:
                    max_num_materials = len(reaction_trees[str(i)]['materials'])
                    final_material_list = reaction_trees[str(i)]['materials']
            dataset.append({
                "product": product, 
                "material_list": final_material_list, 
                "depth": reaction_trees['depth']
            })

    return dataset


def convert_symbols_to_inputs(input_list, output_list, max_length):
    num_samples = len(input_list)
    #input
    input_ids = torch.zeros((num_samples, max_length), device=device, dtype=torch.long)
    input_mask = torch.zeros((num_samples, max_length), device=device)

    #output
    output_ids = torch.zeros((num_samples, max_length), device=device, dtype=torch.long)
    output_mask = torch.zeros((num_samples, max_length), device=device)

    for cnt in range(num_samples):
        input = '^' + input_list[cnt] + '$'
        output = '^' + output_list[cnt] + '$'
        
        for i, symbol in enumerate(input):
            input_ids[cnt, i] = char_to_ix[symbol]
        input_mask[cnt, :len(input)] = 1

        for i in range(len(output)-1):
            output_ids[cnt, i] = char_to_ix[output[i]]
        output_mask[cnt, :len(output)-1] = 1
    return (input_ids, input_mask, output_ids, output_mask)

def get_output_probs(input, res):
    max_length = max(len(input), len(res)) + 5
    test_input_ids, test_input_mask, test_res_ids, test_res_mask = convert_symbols_to_inputs([input], [res], max_length)
    # To Tensor
    test_mutual_mask = get_mutual_mask([test_res_mask, test_input_mask])
    test_input_mask = get_input_mask(test_input_mask)
    test_res_mask = get_output_mask(test_res_mask)

    logits = forward_model(test_input_ids, test_res_ids, test_input_mask, test_res_mask, test_mutual_mask)
    prob = logits[0, len(res), :]
    prob = torch.exp(prob) / torch.sum(torch.exp(prob))
    return prob.detach()

def get_beam(input, beam_size):
    lines = []
    scores = []
    final_beams = []
    object_size = beam_size

    for i in range(object_size):
        lines.append("")
        scores.append(0.0)

    for step in range(args.max_length):
        if step == 0:
            prob = get_output_probs(input, "")
            result = torch.zeros((vocab_size, 2), device=device)
            for i in range(vocab_size):
                result[i, 0] = -torch.log10(prob[i])
                result[i, 1] = i
        else:
            num_candidate = len(lines)
            result = torch.zeros((num_candidate * vocab_size, 2), device=device)
            for i in range(num_candidate):
                prob = get_output_probs(input, lines[i])
                for j in range(vocab_size):
                    result[i*vocab_size+j, 0] = -torch.log10(prob[j]) + scores[i]
                    result[i*vocab_size+j, 1] = i * 100 + j

        ranked_result = result[result[:, 0].argsort()]

        new_beams = []
        new_scores = []
        
        if len(lines) == 0:
            break

        for i in range(object_size):
            symbol = ix_to_char[ranked_result[i, 1].item()%100]
            beam_index = int(ranked_result[i, 1]) // 100

            if symbol == '$':
                added = lines[beam_index] + symbol
                if added != "$":
                    final_beams.append([lines[beam_index] + symbol, ranked_result[i,0]])
                object_size -= 1
            else:
                new_beams.append(lines[beam_index] + symbol)
                new_scores.append(ranked_result[i, 0])

        lines = new_beams
        scores = new_scores

        if len(lines) == 0:
            break

    for i in range(len(final_beams)):
        final_beams[i][1] = final_beams[i][1] / len(final_beams[i][0])

    final_beams = list(sorted(final_beams, key=lambda x:x[1]))
    answer = []
    aim_size = beam_size
    for k in range(len(final_beams)):
        if aim_size == 0:
            break
        output = final_beams[k][0]
        o = output.replace("$", "")
        m = Chem.MolFromSmiles(o)
        if m is not None:
            sms = Chem.MolToSmiles(m)
            answer.append((sms, final_beams[k][1].item()))
            aim_size -= 1
    return answer

def get_forward_result(orginal_input, forward_beam_size):
    results = get_beam(orginal_input, forward_beam_size)
    if len(results) == 0:
        return None
    return results[0]


def check_reactant_is_material(reactant):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] in stock_inchikeys


def check_reactants_are_material(reactants):
    for reactant in reactants:
        if Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] not in stock_inchikeys:
            return False
    return True


def get_candidate_result(task):
    negative_samples_queue = []
    max_depth = task["depth"]
    # Initialization
    answer_set = []
    queue = []
    queue.append({
        "score": 0.0,
        "routes_info": [{"route": [task["product"]], "depth": 0}],  # List of routes information
        "starting_materials": [],
    })
    while True:
        if len(queue) == 0:
            break
        nxt_queue = []
        for item in queue:
            score = item["score"]
            routes_info = item["routes_info"]
            starting_materials = item["starting_materials"]
            first_route_info = routes_info[0]
            first_route, depth = first_route_info["route"], first_route_info["depth"]
            if depth > max_depth:
                continue
            for expansion_solution in proposer.propose(first_route[-1], topk=args.beam_size):
                iter_routes = deepcopy(routes_info)
                iter_routes.pop(0)
                iter_starting_materials = deepcopy(starting_materials)
                expansion_reactants, expansion_score = expansion_solution[0], expansion_solution[1]
                expansion_reactants = sorted(expansion_reactants)
                if check_reactants_are_material(expansion_reactants) and len(iter_routes) == 0:
                    answer_set.append({
                        "score": score+expansion_score,
                        "starting_materials": iter_starting_materials+expansion_reactants,
                        })
                else:
                    for reactant in expansion_reactants:
                        if check_reactant_is_material(reactant):
                            iter_starting_materials.append(reactant)
                        else:
                            iter_routes = [{"route": first_route+[reactant], "depth": depth+1}] + iter_routes
  
                    nxt_queue.append({
                        "score": score+expansion_score,
                        "routes_info": iter_routes,
                        "starting_materials": iter_starting_materials
                    })
        if len(nxt_queue) > 0:
            negative_samples_queue = sorted(nxt_queue, key=lambda x: x["score"])
        queue = sorted(nxt_queue, key=lambda x: x["score"])[:5*args.beam_size]
    
    answer_set = sorted(answer_set, key=lambda x: x["score"])

    ground_truth_material_list = []
    for material_ in task["material_list"]:
        _, cano_material_ = cano_smiles(material_)
        ground_truth_material_list.append(cano_material_)
    ground_truth_material_list = sorted(ground_truth_material_list)

    product_fp = getfp(task["product"])
    ground_truth_material_fp = getfp('.'.join(ground_truth_material_list))

    record_answers = set()
    answer_keys = [Chem.MolToInchiKey(Chem.MolFromSmiles(m))[:14] for m in ground_truth_material_list]
    record_answers.add('.'.join(sorted(answer_keys)))
    final_answer_set = []
    final_answer_set.append({
            "score": 0,
            "forward_sim": 1.0,
            "target_sim": 1.0,
            "total_sim": 2.0,
            "starting_material_list": ground_truth_material_list
        })

    for item in answer_set:
        starting_materials = item["starting_materials"]
        answer_keys = [Chem.MolToInchiKey(Chem.MolFromSmiles(m))[:14] for m in starting_materials]
        if '.'.join(sorted(answer_keys)) not in record_answers:
            record_answers.add('.'.join(sorted(answer_keys)))
        else:
            continue
        
        cano_starting_materials = []
        for material_ in starting_materials:
            _, cano_material_ = cano_smiles(material_)
            cano_starting_materials.append(cano_material_)
        cano_starting_materials = sorted(cano_starting_materials)
        forward_result = get_forward_result('.'.join(cano_starting_materials), args.forward_beam_size)
        if forward_result is None:
            continue
        else:
            forward_product, _ = forward_result
        forward_sim = similarity_metric(product_fp, [getfp(forward_product)])[0]
        target_sim = similarity_metric(ground_truth_material_fp, [getfp('.'.join(cano_starting_materials))])[0]

        final_answer_set.append({
            "score": item["score"],
            "forward_sim": forward_sim,
            "target_sim": target_sim,
            "total_sim": forward_sim+target_sim,
            "starting_material_list": cano_starting_materials
        })
        if len(final_answer_set) == args.candidate_size:
            break
        
    
    if len(final_answer_set) < args.candidate_size:
        print(task['product'], len(final_answer_set))
        for item in negative_samples_queue:
            routes_info = item["routes_info"]
            starting_materials = []
            for route_item in routes_info:
                starting_materials.append(route_item["route"][-1])
            answer_keys = [Chem.MolToInchiKey(Chem.MolFromSmiles(m))[:14] for m in starting_materials]
            if '.'.join(sorted(answer_keys)) not in record_answers:
                record_answers.add('.'.join(sorted(answer_keys)))
            else:
                continue
            
            cano_starting_materials = []
            for material_ in starting_materials:
                _, cano_material_ = cano_smiles(material_)
                cano_starting_materials.append(cano_material_)
            cano_starting_materials = sorted(cano_starting_materials)
            forward_result = get_forward_result('.'.join(cano_starting_materials), args.forward_beam_size)
            if forward_result is None:
                continue
            else:
                forward_product, _ = forward_result
            forward_sim = similarity_metric(product_fp, [getfp(forward_product)])[0]
            target_sim = similarity_metric(ground_truth_material_fp, [getfp('.'.join(cano_starting_materials))])[0]

            final_answer_set.append({
                "score": item["score"],
                "forward_sim": forward_sim,
                "target_sim": target_sim,
                "total_sim": forward_sim+target_sim,
                "starting_material_list": cano_starting_materials
            })
            if len(final_answer_set) == args.candidate_size:
                break

    final_answer_set = sorted(final_answer_set, key=lambda x: x["total_sim"], reverse=True)
    print(task['product'], "final size: ", len(final_answer_set))
    if len(final_answer_set) != 1:
        candidate_result[task["product"]] = []
        for item in final_answer_set:
            candidate = '.'.join(item["starting_material_list"])
            candidate_result[task["product"]].append(candidate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("dfs_bfs_search.py")
    parser.add_argument("--beam_size", help="beam size", type=int, default=50)
    parser.add_argument("--candidate_size", help="beam size", type=int, default=10)
    parser.add_argument("--forward_beam_size", help="beam size", type=int, default=1)
    parser.add_argument('--max_length', type=int, default=350)

    args = parser.parse_args()
    device_neuralsym = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    DATA_FOLDER = Path(__file__).resolve().parent / 'data'
    CHECKPOINT_FOLDER = Path(__file__).resolve().parent / 'checkpoint'

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    infer_config = {
        'templates_file': 'training_templates',
        'min_freq': 1,
        'expt_name': 'Highway_42_depth0_dim300_lr1e3_stop2_fac30_pat1',
        'hidden_size': 300,
        'depth': 0,
        'orig_fp_size': 1000000,
        'final_fp_size': 32681,
        'radius': 2,
    }


    proposer = Proposer(infer_config)

    stock = pd.read_hdf('zinc_stock_17_04_20.hdf5', key="table")  
    stockinchikey_list = stock.inchi_key.values
    stock_inchikeys = set([x[:14] for x in stockinchikey_list])

    similarity_metric = DataStructs.BulkTanimotoSimilarity # BulkDiceSimilarity or BulkTanimotoSimilarity
    similarity_label = 'Tanimoto'
    getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=True)

    chars = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"
    vocab_size = len(chars)

    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    stock = pd.read_hdf('zinc_stock_17_04_20.hdf5', key="table")  
    stockinchikey_list = stock.inchi_key.values
    stock_inchikeys = set([x[:14] for x in stockinchikey_list])
    config = ForwardTransformerConfig(vocab_size=vocab_size,
                                      embedding_size=64,
                                      hidden_size=512,
                                      num_hidden_layers=6,
                                      num_attention_heads=8,
                                      intermediate_size=1024,
                                      hidden_dropout_prob=0.1)
    forward_model = ForwardTransformer(config)
    checkpoint = torch.load("forward_model.pkl")
    if isinstance(checkpoint, torch.nn.DataParallel):
        checkpoint = checkpoint.module
    forward_model.load_state_dict(checkpoint.state_dict())

    forward_model.to(device)
    forward_model.eval()

    candidate_result = {}
    tasks = load_dataset("train")
    for epoch in trange(0, len(tasks)):
        get_candidate_result(tasks[epoch])

    with open('candidate_dataset.json', 'w') as f:
        f.write(json.dumps(candidate_result, indent=4))

