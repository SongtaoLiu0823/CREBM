import argparse
import random
import numpy as np
import torch
from tqdm import trange
from forward_preprocess import get_vocab_size, get_char_to_ix, get_ix_to_char, get_dataset
from forward_model import ForwardTransformerConfig, ForwardTransformer, get_input_mask, get_output_mask, get_mutual_mask
from rdkit import Chem
from rdkit.rdBase import DisableLog

DisableLog('rdApp.warning')


def convert_symbols_to_inputs(input_list, output_list, max_length):
    num_samples = len(input_list)
    #input
    input_ids = torch.zeros((num_samples, max_length), device=device, dtype=torch.long)
    input_mask = torch.zeros((num_samples, max_length), device=device)

    #output
    output_ids = torch.zeros((num_samples, max_length), device=device, dtype=torch.long)
    output_mask = torch.zeros((num_samples, max_length), device=device)

    for cnt in range(num_samples):
        input_ = '^' + input_list[cnt] + '$'
        output_ = '^' + output_list[cnt] + '$'
        
        for i, symbol in enumerate(input_):
            input_ids[cnt, i] = char_to_ix[symbol]
        input_mask[cnt, :len(input_)] = 1

        for i in range(len(output_)-1):
            output_ids[cnt, i] = char_to_ix[output_[i]]
        output_mask[cnt, :len(output_)-1] = 1
    return (input_ids, input_mask, output_ids, output_mask)


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


def get_output_probs(test_input, res):
    test_input_ids, test_input_mask, test_res_ids, test_res_mask = convert_symbols_to_inputs([test_input], [res], args.max_length)
    # To Tensor
    test_mutual_mask = get_mutual_mask([test_res_mask, test_input_mask])
    test_input_mask = get_input_mask(test_input_mask)
    test_res_mask = get_output_mask(test_res_mask)

    logits = predict_model(test_input_ids, test_res_ids, test_input_mask, test_res_mask, test_mutual_mask)
    prob = logits[0, len(res), :] / args.temperature
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


def get_prediction_result(reaction_input, ground_truth_output):
    _, reaction_input = cano_smiles(reaction_input)
    ground_truth_key = Chem.MolToInchiKey(Chem.MolFromSmiles(ground_truth_output))
    results = get_beam(reaction_input, args.beam_size)
    for rank, solution in enumerate(results):
        predict_output, _ = solution[0], solution[1]
        answer_key = Chem.MolToInchiKey(Chem.MolFromSmiles(predict_output))
        if answer_key[:14] == ground_truth_key[:14]:
            return rank
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max_length', type=int, default=400, help='The max length of a molecule.')
    parser.add_argument('--embedding_size', type=int, default=64, help='The size of embeddings')
    parser.add_argument('--hidden_size', type=int, default=512, help='The size of hidden units')
    parser.add_argument('--num_hidden_layers', type=int, default=6, help='Number of layers in encoder\'s module. Default 6.')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='Number of attention heads. Default 8.')
    parser.add_argument('--intermediate_size', type=int, default=1024, help='The size of hidden units of position-wise layer.')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--temperature', type=float, default=1.2, help='Temperature for decoding. Default 1.2')
    parser.add_argument('--beam_size', type=int, default=10, help='Beams size. Default 5. Must be 1 meaning greedy search or greater or equal 5.')

    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = ForwardTransformerConfig(vocab_size=get_vocab_size(),
                                      embedding_size=args.embedding_size,
                                      hidden_size=args.hidden_size,
                                      num_hidden_layers=args.num_hidden_layers,
                                      num_attention_heads=args.num_attention_heads,
                                      intermediate_size=args.intermediate_size,
                                      hidden_dropout_prob=args.hidden_dropout_prob)

    predict_model = ForwardTransformer(config)
    checkpoint = torch.load("models/forward_model.pkl")
    if isinstance(checkpoint, torch.nn.DataParallel):
        checkpoint = checkpoint.module
    predict_model.load_state_dict(checkpoint.state_dict())

    predict_model.to(device)
    predict_model.eval()

    char_to_ix = get_char_to_ix()
    ix_to_char = get_ix_to_char()
    vocab_size = get_vocab_size()


    overall_result = np.zeros((args.beam_size, 2))
    test_input_list, test_output_list = get_dataset('test')
    for epoch in trange(0, len(test_input_list)):
        test_input = test_input_list[epoch]
        ground_truth_output = test_output_list[epoch]
        rank = get_prediction_result(test_input, ground_truth_output)
        overall_result[:, 1] += 1
        if rank is not None:
            overall_result[rank:, 0] += 1
        if (epoch + 1) % 10 == 0:
            print("overall_result: ", overall_result, 100 * overall_result[:, 0] / overall_result[:, 1])
    print("overall_result: ", overall_result, 100 * overall_result[:, 0] / overall_result[:, 1])
