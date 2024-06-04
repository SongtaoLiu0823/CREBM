import os
import numpy as np
import torch
import pandas as pd
import math
import json
import argparse

from copy import deepcopy
from rdkit import Chem
from tqdm import trange
from gln.common.cmd_args import cmd_args
from gln.common.consts import DEVICE
from gln.test.model_inference import RetroGLN
from gln.reward_model import RewardTransformerConfig, RewardTransformer, get_input_mask, get_output_mask, get_mutual_mask

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


def get_inference_answer(smiles, beam_size):
    pred_struct = model.run(smiles, 5*beam_size, 5*beam_size, rxn_type='UNK')
    if pred_struct is None:
        return []
    reactants_list = pred_struct['reactants']
    scores_list = pred_struct['scores']
    answer = []
    aim_size = beam_size
    for i in range(len(reactants_list)):
        if aim_size == 0:
            break
        reactants = reactants_list[i].split('.')
        score = scores_list[i]
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
            try:
                answer.append([sorted(list(sms)), -math.log10(score)]) 
            except:
                answer.append([sorted(list(sms)), -math.log10(score+1e-10)])
            aim_size -= 1

    return answer


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
            materials_list = []
            for i in range(1, int(reaction_trees['num_reaction_trees'])+1):
                materials_list.append(reaction_trees[str(i)]['materials'])
            dataset.append({
                "product": product,
                "targets": materials_list, 
                "depth": reaction_trees['depth']
            })

    return dataset

def convert_symbols_to_inputs(input_list, output_list, max_length):
    num_samples = len(input_list)
    #input
    input_ids = np.zeros((num_samples, max_length))
    input_mask = np.zeros((num_samples, max_length))

    #output
    output_ids = np.zeros((num_samples, max_length))
    output_mask = np.zeros((num_samples, max_length))

    #for output
    token_ids = np.zeros((num_samples, max_length))
    token_mask = np.zeros((num_samples, max_length))

    for cnt in range(num_samples):
        input_ = '^' + input_list[cnt] + '$'
        output_ = '^' + output_list[cnt] + '$'
        
        for i, symbol in enumerate(input_):
            input_ids[cnt, i] = char_to_ix[symbol]
        input_mask[cnt, :len(input_)] = 1

        for i in range(len(output_)-1):
            output_ids[cnt, i] = char_to_ix[output_[i]]
            token_ids[cnt, i] = char_to_ix[output_[i+1]]
            if i != len(output_)-2:
                token_mask[cnt, i] = 1
        output_mask[cnt, :len(output_)-1] = 1
    return (input_ids, input_mask, output_ids, output_mask, token_ids, token_mask)

def get_rerank_scores(input_list, output_list):
    if input_list:
        longest_input = max(input_list, key=len)
        max_length_input = len(longest_input)
    else:
        max_length_input = 0
    if output_list:
        longest_output = max(output_list, key=len)
        max_length_output = len(longest_output)
    else:
        max_length_output = 0
    max_length_reward = max(max_length_input, max_length_output) + 2
    (input_ids, 
    input_mask, 
    output_ids, 
    output_mask, 
    token_ids,
    token_mask) = convert_symbols_to_inputs(input_list, output_list, max_length_reward)

    input_ids = torch.LongTensor(input_ids).to(DEVICE)
    input_mask = torch.FloatTensor(input_mask).to(DEVICE)
    output_ids = torch.LongTensor(output_ids).to(DEVICE)
    output_mask = torch.FloatTensor(output_mask).to(DEVICE)
    token_ids = torch.LongTensor(token_ids).to(DEVICE)
    token_mask = torch.FloatTensor(token_mask).to(DEVICE)
    mutual_mask = get_mutual_mask([output_mask, input_mask])
    input_mask = get_input_mask(input_mask)
    output_mask = get_output_mask(output_mask)
    logits = reward_model(input_ids, output_ids, input_mask, output_mask, mutual_mask)
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=token_ids.unsqueeze(2)).squeeze(2)

    all_logps = (per_token_logps * token_mask).sum(-1) / token_mask.sum(-1)
    return all_logps

def check_reactant_is_material(reactant):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] in stock_inchikeys


def check_reactants_are_material(reactants):
    for reactant in reactants:
        if Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] not in stock_inchikeys:
            return False
    return True


def get_route_result(task):
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
            for expansion_solution in get_inference_answer(first_route[-1], local_args.beam_size):
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
        queue = sorted(nxt_queue, key=lambda x: x["score"])[:local_args.beam_size]
            
    answer_set = sorted(answer_set, key=lambda x: x["score"])
    record_answers = set()
    final_answer_set = []
    rerank_input_list = []
    rerank_output_list = []
    for item in answer_set:
        score = item["score"]
        starting_materials = item["starting_materials"]

        cano_starting_materials = []
        for material_ in starting_materials:
            _, cano_material_ = cano_smiles(material_)
            cano_starting_materials.append(cano_material_)

        answer_keys = [Chem.MolToInchiKey(Chem.MolFromSmiles(m))[:14] for m in starting_materials]
        if '.'.join(sorted(answer_keys)) not in record_answers:
            record_answers.add('.'.join(sorted(answer_keys)))
            final_answer_set.append({
                "score": score,
                "answer_keys": answer_keys
            })
            rerank_input_list.append(task['product'])
            rerank_output_list.append('.'.join(sorted(cano_starting_materials)))

    rerank_scores = get_rerank_scores(rerank_input_list, rerank_output_list)
    for i, score_ in enumerate(rerank_scores):
        final_answer_set[i]["rerank_score"] = -score_.item()
        final_answer_set[i]["total_score"] = -local_args.alpha*score_.item() + final_answer_set[i]["score"]
    final_answer_set = sorted(final_answer_set, key=lambda x: x["total_score"])[:local_args.beam_size]

    # Calculate answers
    ground_truth_keys_list = [
        set([
            Chem.MolToInchiKey(Chem.MolFromSmiles(target))[:14] for target in targets
        ]) for targets in task["targets"]
    ]
    for rank, answer in enumerate(final_answer_set):
        answer_keys = set(answer["answer_keys"])
        for ground_truth_keys in ground_truth_keys_list:
            if ground_truth_keys == answer_keys:
                return max_depth, rank
    
    return max_depth, None


if __name__ == "__main__":
    cmd_opt = argparse.ArgumentParser(description='Argparser for search')
    cmd_opt.add_argument('-epoch_for_search', default=100, type=int, help='model for search')
    cmd_opt.add_argument("-beam_size", help="beam size", type=int, default=5)
    cmd_opt.add_argument("-alpha", type=float, default=1.0)
    local_args, _ = cmd_opt.parse_known_args()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    model_dump = os.path.join(cmd_args.save_dir, 'model-%d.dump' % local_args.epoch_for_search)
    model = RetroGLN(model_dump)
    model.gln.to(DEVICE)

    stock = pd.read_hdf('zinc_stock_17_04_20.hdf5', key="table")  
    stockinchikey_list = stock.inchi_key.values
    stock_inchikeys = set([x[:14] for x in stockinchikey_list])

    chars = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"
    vocab_size = len(chars)
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    config = RewardTransformerConfig(vocab_size=vocab_size,
                                     embedding_size=64,
                                     hidden_size=512,
                                     num_hidden_layers=6,
                                     num_attention_heads=8,
                                     intermediate_size=1024,
                                     hidden_dropout_prob=0.1)
    reward_model = RewardTransformer(config)
    checkpoint = torch.load("reward_model%.pkl")
    reward_model.load_state_dict(checkpoint.state_dict())
    reward_model.to(DEVICE)
    reward_model.eval()

    overall_result = np.zeros((local_args.beam_size, 2))
    depth_hit = np.zeros((2, 15, local_args.beam_size))
    tasks = load_dataset("test")
    for epoch in trange(0, len(tasks)):
        max_depth, rank = get_route_result(tasks[epoch])
        overall_result[:, 1] += 1
        depth_hit[1, max_depth, :] += 1
        if rank is not None:
            overall_result[rank:, 0] += 1
            depth_hit[0, max_depth, rank:] += 1

    print("overall_result: ", overall_result, 100 * overall_result[:, 0] / overall_result[:, 1])
    print("depth_hit: ", depth_hit, 100 * depth_hit[0, :, :] / depth_hit[1, :, :])
