import numpy as np
import pandas as pd
import torch
import os
import json
import argparse
import yaml

from tqdm import trange
from copy import deepcopy
from rdkit import RDLogger, Chem

from seq_graph_retro.utils.edit_mol import generate_reac_set
from seq_graph_retro.models import EditLGSeparate
from seq_graph_retro.search import BeamSearch
from reward_model import RewardTransformerConfig, RewardTransformer, get_input_mask, get_output_mask, get_mutual_mask
lg = RDLogger.logger()
lg.setLevel(4)

try:
    ROOT_DIR = os.environ["SEQ_GRAPH_RETRO"]
    DATA_DIR = os.path.join(ROOT_DIR, "datasets", "uspto-50k")
    EXP_DIR = os.path.join(ROOT_DIR, "experiments")
    REWARD_DIR = os.path.join(ROOT_DIR, "scripts/eval")

except KeyError:
    ROOT_DIR = "./"
    DATA_DIR = os.path.join(ROOT_DIR, "datasets", "uspto-50k")
    EXP_DIR = os.path.join(ROOT_DIR, "local_experiments")

DEFAULT_TEST_FILE = f"{DATA_DIR}/canonicalized_test.csv"


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


def canonicalize_prod(pcanon):
    pmol = Chem.MolFromSmiles(pcanon)
    [atom.SetAtomMapNum(atom.GetIdx()+1) for atom in pmol.GetAtoms()]
    p = Chem.MolToSmiles(pmol)
    return p


def load_edits_model(args):
    edits_step = args.edits_step
    if edits_step is None:
        edits_step = "best_model"

    if "run" in args.edits_exp:
        # This addition because some of the new experiments were run using wandb
        edits_loaded = torch.load(os.path.join(args.exp_dir, "wandb", args.edits_exp, "files", edits_step + ".pt"), map_location=device)
        with open(f"{args.exp_dir}/wandb/{args.edits_exp}/files/config.yaml", "r") as f:
            tmp_loaded = yaml.load(f, Loader=yaml.FullLoader)

        model_name = tmp_loaded['model']['value']

    else:
        edits_loaded = torch.load(os.path.join(args.exp_dir, args.edits_exp,
                                  "checkpoints", edits_step + ".pt"),
                                  map_location=device)
        model_name = args.edits_exp.split("_")[0]

    return edits_loaded, model_name


def load_lg_model(args):
    lg_step = args.lg_step
    if lg_step is None:
        lg_step = "best_model"

    if "run" in args.lg_exp:
        # This addition because some of the new experiments were run using wandb
        lg_loaded = torch.load(os.path.join(args.exp_dir, "wandb", args.lg_exp, "files", lg_step + ".pt"), map_location=device)
        with open(f"{args.exp_dir}/wandb/{args.lg_exp}/files/config.yaml", "r") as f:
            tmp_loaded = yaml.load(f, Loader=yaml.FullLoader)

        model_name = tmp_loaded['model']['value']

    else:
        lg_loaded = torch.load(os.path.join(args.exp_dir, args.lg_exp,
                               "checkpoints", lg_step + ".pt"),
                                map_location=device)
        model_name = args.lg_exp.split("_")[0]

    return lg_loaded, model_name


def load_dataset(split):
    file_name = "%s/%s_dataset.json" % (DATA_DIR, split)
    file_name = os.path.expanduser(file_name)
    dataset = [] # (product_smiles, materials_smiles, depth)
    with open(file_name, 'r') as f:
        _dataset = json.load(f)
        for _, reaction_trees in _dataset.items():
            product = reaction_trees['1']['retro_routes'][0][0].split('>')[0]
            product_mol = Chem.MolFromInchi(Chem.MolToInchi(Chem.MolFromSmiles(product)))
            product = Chem.MolToSmiles(product_mol)
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

    input_ids = torch.LongTensor(input_ids).to(device)
    input_mask = torch.FloatTensor(input_mask).to(device)
    output_ids = torch.LongTensor(output_ids).to(device)
    output_mask = torch.FloatTensor(output_mask).to(device)
    token_ids = torch.LongTensor(token_ids).to(device)
    token_mask = torch.FloatTensor(token_mask).to(device)
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


def get_prediction(p):
    p = canonicalize_prod(p)
    rxn_class = None
    try:
        answer = []
        if lg_toggles.get("use_rxn_class", False):
            top_k_nodes = beam_model.run_search(p, max_steps=6, rxn_class=rxn_class)
        else:
            top_k_nodes = beam_model.run_search(p, max_steps=6)

        for beam_idx, node in enumerate(top_k_nodes):
            
            if len(answer) == args.beam_size:
                return answer

            pred_edit = node.edit
            pred_label = node.lg_groups
            score = -node.prob

            if isinstance(pred_edit, list):
                pred_edit = pred_edit[0]
            try:
                pred_set = generate_reac_set(p, pred_edit, pred_label, verbose=False)
                num_valid_reactant = 0
                sms = set()
                for r in pred_set:
                    m = Chem.MolFromSmiles(r)
                    if m is not None:
                        num_valid_reactant += 1
                        sms.add(Chem.MolToSmiles(m))
                if num_valid_reactant != len(pred_set):
                    continue
                if len(sms):
                    answer.append([sorted(list(sms)), score])

            except BaseException as e:
                print(e, flush=True)
                pred_set = None

        return answer[:args.beam_size]
    
    except Exception as e:
        return []


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
            for expansion_solution in get_prediction(first_route[-1]):
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
        queue = sorted(nxt_queue, key=lambda x: x["score"])[:args.beam_size]
            
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
        final_answer_set[i]["total_score"] = -args.alpha*score_.item() + final_answer_set[i]["score"]
    final_answer_set = sorted(final_answer_set, key=lambda x: x["total_score"])[:args.beam_size]

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


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=DATA_DIR, help="Data directory")
parser.add_argument("--exp_dir", default=EXP_DIR, help="Experiments directory.")
parser.add_argument("--test_file", default=DEFAULT_TEST_FILE, help="Test file.")
parser.add_argument("--edits_exp", default="SingleEdit_21-03-2020--20-33-05", help="Name of edit prediction experiment.")
parser.add_argument("--edits_step", default=None, help="Checkpoint for edit prediction experiment.")
parser.add_argument("--lg_exp", default="LGClassifier_02-04-2020--02-06-17", help="Name of synthon completion experiment.")
parser.add_argument("--lg_step", default=None, help="Checkpoint for synthon completion experiment.")
parser.add_argument("--beam_width", default=10, type=int, help="Beam width")
parser.add_argument("--use_rxn_class", action='store_true', help="Whether to use reaction class.")
parser.add_argument("--rxn_class_acc", action="store_true", help="Whether to print reaction class accuracy.")
parser.add_argument('--beam_size', type=int, default=5, help='Beams size. Default 5. Must be 1 meaning greedy search or greater or equal 5.')
parser.add_argument("--alpha", type=float, default=1.0)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_df = pd.read_csv(args.test_file)

edits_loaded, edit_net_name = load_edits_model(args)
lg_loaded, lg_net_name = load_lg_model(args)

edits_config = edits_loaded["saveables"]
lg_config = lg_loaded['saveables']
lg_toggles = lg_config['toggles']

if 'tensor_file' in lg_config:
    if not os.path.isfile(lg_config['tensor_file']):
        if not lg_toggles.get("use_rxn_class", False):
            tensor_file = os.path.join(args.data_dir, "train/h_labels/without_rxn/lg_inputs.pt")
        else:
            tensor_file = os.path.join(args.data_dir, "train/h_labels/with_rxn/lg_inputs.pt")
        lg_config['tensor_file'] = tensor_file

rm = EditLGSeparate(edits_config=edits_config, lg_config=lg_config, edit_net_name=edit_net_name,
                    lg_net_name=lg_net_name, device=device)
rm.load_state_dict(edits_loaded['state'], lg_loaded['state'])
rm.to(device)
rm.eval()

beam_model = BeamSearch(model=rm, beam_width=args.beam_width, max_edits=1)

stock = pd.read_hdf('%s/zinc_stock_17_04_20.hdf5' %DATA_DIR, key="table")  
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
checkpoint = torch.load("%s/reward_model.pkl"%REWARD_DIR)
reward_model.load_state_dict(checkpoint.state_dict())
reward_model.to(device)
reward_model.eval()

tasks = load_dataset('test')
overall_result = np.zeros((args.beam_size, 2))
depth_hit = np.zeros((2, 15, args.beam_size))
for epoch in trange(0, len(tasks)):
    max_depth, rank = get_route_result(tasks[epoch])
    overall_result[:, 1] += 1
    depth_hit[1, max_depth, :] += 1
    if rank is not None:
        overall_result[rank:, 0] += 1
        depth_hit[0, max_depth, rank:] += 1

print("overall_result: ", overall_result, 100 * overall_result[:, 0] / overall_result[:, 1])
print("depth_hit: ", depth_hit, 100 * depth_hit[0, :, :] / depth_hit[1, :, :])
