import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import json

from tqdm import tqdm, trange
from reward_model import RewardTransformerConfig, RewardTransformer, get_input_mask, get_output_mask, get_mutual_mask
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler
from rdkit.rdBase import DisableLog

DisableLog('rdApp.warning')

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--max_length', type=int, default=310, help='The max length of a molecule.')
parser.add_argument('--embedding_size', type=int, default=64, help='The size of embeddings')
parser.add_argument('--hidden_size', type=int, default=512, help='The size of hidden units')
parser.add_argument('--num_hidden_layers', type=int, default=6, help='Number of layers in encoder\'s module. Default 6.')
parser.add_argument('--num_attention_heads', type=int, default=8, help='Number of attention heads. Default 8.')
parser.add_argument('--intermediate_size', type=int, default=1024, help='The size of hidden units of position-wise layer.')
parser.add_argument('--hidden_dropout_prob', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train.')
parser.add_argument("--batch_size", default=64, type=int, help="Total batch size for training.")
parser.add_argument("--warmup", default=1000, type=float)
parser.add_argument("--lr", default=5e-5, type=float)

args = parser.parse_args()

class PairedDataset(Dataset):
    def __init__(self, accept_dataset, reject_dataset):
        assert len(accept_dataset) == len(reject_dataset)
        self.accept_dataset = accept_dataset
        self.reject_dataset = reject_dataset

    def __len__(self):
        return len(self.accept_dataset)

    def __getitem__(self, idx):
        accept_data = self.accept_dataset[idx]
        reject_data = self.reject_dataset[idx]
        return accept_data, reject_data

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpu = torch.cuda.device_count()

chars = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"
vocab_size = len(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

config = RewardTransformerConfig(vocab_size=vocab_size,
                                 embedding_size=args.embedding_size,
                                 hidden_size=args.hidden_size,
                                 num_hidden_layers=args.num_hidden_layers,
                                 num_attention_heads=args.num_attention_heads,
                                 intermediate_size=args.intermediate_size,
                                 hidden_dropout_prob=args.hidden_dropout_prob)


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

    for cnt in trange(num_samples):
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

with open("candidate_dataset.json", 'r') as f:
    candidate_new_dataset = json.load(f)

accept_input_list = []
accept_output_list = []
reject_input_list = []
reject_output_list = []

max_length = 0
for product, cano_starting_material_list in tqdm(candidate_new_dataset.items()):
    for i in range(len(cano_starting_material_list)-1):
        for j in range(i+1, len(cano_starting_material_list)):
            accept_input = product
            reject_input = product
            accept_output = cano_starting_material_list[i]
            reject_output = cano_starting_material_list[j]
            max_length = max(max_length, len(product), len(accept_output), len(reject_output))
            accept_input_list.append(accept_input)
            accept_output_list.append(accept_output)
            reject_input_list.append(reject_input)
            reject_output_list.append(reject_output)

(accept_input_ids, 
accept_input_mask, 
accept_output_ids, 
accept_output_mask, 
accept_token_ids,
accept_token_mask) = convert_symbols_to_inputs(accept_input_list, accept_output_list, args.max_length)

accept_input_ids = torch.LongTensor(accept_input_ids).to(device)
accept_input_mask = torch.FloatTensor(accept_input_mask).to(device)
accept_output_ids = torch.LongTensor(accept_output_ids).to(device)
accept_output_mask = torch.FloatTensor(accept_output_mask).to(device)
accept_token_ids = torch.LongTensor(accept_token_ids).to(device)
accept_token_mask = torch.FloatTensor(accept_token_mask).to(device)

accept_data = TensorDataset(accept_input_ids, 
                            accept_input_mask, 
                            accept_output_ids, 
                            accept_output_mask, 
                            accept_token_ids, 
                            accept_token_mask)

(reject_input_ids, 
reject_input_mask, 
reject_output_ids, 
reject_output_mask, 
reject_token_ids,
reject_token_mask) = convert_symbols_to_inputs(reject_input_list, reject_output_list, args.max_length)

reject_input_ids = torch.LongTensor(reject_input_ids).to(device)
reject_input_mask = torch.FloatTensor(reject_input_mask).to(device)
reject_output_ids = torch.LongTensor(reject_output_ids).to(device)
reject_output_mask = torch.FloatTensor(reject_output_mask).to(device)
reject_token_ids = torch.LongTensor(reject_token_ids).to(device)
reject_token_mask = torch.FloatTensor(reject_token_mask).to(device)

reject_data = TensorDataset(reject_input_ids, 
                            reject_input_mask, 
                            reject_output_ids, 
                            reject_output_mask,
                            reject_token_ids,
                            reject_token_mask)

paired_dataset = PairedDataset(accept_data, reject_data)
paired_sampler = RandomSampler(paired_dataset)
paired_dataloader = DataLoader(paired_dataset, sampler=paired_sampler, batch_size=args.batch_size)
    
model = RewardTransformer(config)
checkpoint = torch.load("pretrain_reward_models/pretrain_reward_model.pkl")
if isinstance(checkpoint, torch.nn.DataParallel):
    checkpoint = checkpoint.module
model.load_state_dict(checkpoint.state_dict())
model.to(device)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

optimizer = torch.optim.Adam(model.parameters(), args.lr)
num_training_steps = len(paired_dataloader) * args.epochs
# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warmup,
    num_training_steps=num_training_steps,
)

for epoch in trange(1, int(args.epochs)+1, desc="Epoch"):
    paired_dataloader = tqdm(paired_dataloader, desc='Iteration')
    total_step = 0
    total_loss = 0
    for accept_batch, reject_batch in paired_dataloader:
        model.train()
        optimizer.zero_grad()
        accept_input_ids, accept_input_mask, accept_output_ids, accept_output_mask, accept_token_ids, accept_token_mask = accept_batch
        reject_input_ids, reject_input_mask, reject_output_ids, reject_output_mask, reject_token_ids, reject_token_mask = reject_batch
        
        input_ids = torch.cat((accept_input_ids, reject_input_ids), dim=0)
        input_mask = torch.cat((accept_input_mask, reject_input_mask), dim=0)
        output_ids = torch.cat((accept_output_ids, reject_output_ids), dim=0)
        output_mask = torch.cat((accept_output_mask, reject_output_mask), dim=0)
        token_ids = torch.cat((accept_token_ids, reject_token_ids), dim=0)
        token_mask = torch.cat((accept_token_mask, reject_token_mask), dim=0)

        mutual_mask = get_mutual_mask([output_mask, input_mask])
        input_mask = get_input_mask(input_mask)
        output_mask = get_output_mask(output_mask)
        logits = model(input_ids, output_ids, input_mask, output_mask, mutual_mask)
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=token_ids.unsqueeze(2)).squeeze(2)

        all_logps = (per_token_logps * token_mask).sum(-1) / token_mask.sum(-1)
        b_size = int(all_logps.shape[0] / 2)
        chosen_rewards, rejected_rewards = all_logps[:b_size], all_logps[b_size:]
        loss_rank = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        loss_rank.backward()
        optimizer.step()
        scheduler.step()

        total_step += chosen_rewards.shape[0]
        total_loss += loss_rank.item() * chosen_rewards.shape[0]

        paired_dataloader.set_description(f"loss={total_loss/total_step:.4f}")
        paired_dataloader.refresh()

    torch.save(model, "reward_models/epoch_%s_reward_model.pkl"%epoch)

