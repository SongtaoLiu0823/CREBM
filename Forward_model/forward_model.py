import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from forward_preprocess import *


def get_position_embedding(x, embedding_size):
    mask = torch.unsqueeze(torch.arange(1, x.shape[1]+1, device=x.device).float(), dim=-1)
    bins = torch.unsqueeze(2 * torch.arange(embedding_size // 2, device=x.device).float(), dim=0)

    evens = torch.matmul(mask, 1.0 / torch.pow(10000.0, bins / embedding_size))
    odds = evens.clone()

    evens = torch.sin(evens)
    odds = torch.cos(odds)

    pos = torch.stack([evens, odds], dim=2).reshape(x.shape[1], embedding_size)
    return pos.expand(x.shape[0], pos.shape[0], pos.shape[1])


def get_attention_mask(mask, num_attention_heads):
    attention_mask =  mask.expand(num_attention_heads, mask.shape[0], mask.shape[1], mask.shape[2]).permute(1, 0, 2, 3)
    attention_mask = (1.0 - attention_mask) * -10000.0
    return attention_mask


def get_input_mask(x):

    length = x.shape[1]
    rank = torch.ones(size=(1, length), device=x.device)
    y = torch.unsqueeze(x, dim=-1)

    mask = torch.matmul(y, rank)
    return mask.permute(0,2,1)


def get_mutual_mask(x):

    output = x[0]
    input = x[1]

    length = output.shape[1]
    rank = torch.ones(size=(1, length), device=input.device)
    y = torch.unsqueeze(input, dim=-1)

    mask = torch.matmul(y, rank)
    return mask.permute(0,2,1)


def get_output_mask(x):

    t = torch.ones(size=(x.shape[0], x.shape[1], x.shape[1]), device=x.device)
    tri = torch.tril(t, diagonal=0)

    rank = torch.ones(size=(1, x.shape[1]), device=x.device)
    y = torch.unsqueeze(x, dim=-1)

    mask = torch.matmul(y, rank)
    return tri * mask.permute(0,2,1)


class ForwardTransformerConfig(object):
    """Configuration class to store the configuration of a `TransformerConfig`.
    """
    def __init__(self,
                 vocab_size=get_vocab_size(),
                 embedding_size=64,
                 hidden_size=640,
                 num_hidden_layers=3,
                 num_attention_heads=10,
                 intermediate_size=512,
                 hidden_dropout_prob=0.1):
        """ Constructs Config. """
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output


class LayerNorm(nn.Module):
    def __init__(self, config, eps=1e-12):
        """ Construct a layernorm module in the TF style (epsilon inside the square root). """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.embedding_size))
        self.beta = nn.Parameter(torch.zeros(config.embedding_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """ The embedding module from word and position embeddings. """
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.embedding_size = config.embedding_size
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        position_embeddings = get_position_embedding(input_ids, self.embedding_size)
        embeddings = self.word_embeddings(input_ids) + position_embeddings

        return self.dropout(embeddings)


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)

        self.query = nn.Linear(config.embedding_size, config.hidden_size)
        self.key = nn.Linear(config.embedding_size, config.hidden_size)
        self.value = nn.Linear(config.embedding_size, config.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, inputs, attention_mask):
        query_layer = self.transpose_for_scores(self.query(inputs[0]))
        key_layer = self.transpose_for_scores(self.key(inputs[1]))
        value_layer = self.transpose_for_scores(self.value(inputs[2]))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (context_layer.size(2)*context_layer.size(3),)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class EncoderLayer(nn.Module):
    """ Encoder Layer """
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(config)
        self.addnorm_dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.addnorm_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.addnorm_layernorm = LayerNorm(config, eps=1e-12)

        self.ffn_dense1 = nn.Linear(config.embedding_size, config.intermediate_size)
        self.ffn_dense2 = nn.Linear(config.intermediate_size, config.embedding_size)
        self.ffn_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = LayerNorm(config, eps=1e-12)

    def forward(self, hidden_states, attention_mask):
        # self attention
        attention_output = self.attention((hidden_states, hidden_states, hidden_states), attention_mask)
        dense_output = self.addnorm_dense(attention_output)
        drop_output = self.addnorm_dropout(dense_output)
        final_attention_output = self.addnorm_layernorm(drop_output+hidden_states)

        # position-wise
        ff_output = self.ffn_dropout(self.ffn_dense2(F.relu(self.ffn_dense1(final_attention_output))))
        hidden_states = self.layernorm(ff_output+final_attention_output)
        
        return hidden_states


class Encoder(nn.Module):
    """ Encoder """
    def __init__(self, config):
        super(Encoder, self).__init__()
        layer = EncoderLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class DecoderLayer(nn.Module):
    """ Decoder Layer """
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(config)
        self.selfatt_addnorm_dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.selfatt_addnorm_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.selfatt_addnorm_layernorm = LayerNorm(config, eps=1e-12)

        self.multual_attention = MultiHeadAttention(config)
        self.multualatt_addnorm_dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.multualatt_addnorm_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.multualatt_addnorm_layernorm = LayerNorm(config, eps=1e-12)


        self.ffn_dense1 = nn.Linear(config.embedding_size, config.intermediate_size)
        self.ffn_dense2 = nn.Linear(config.intermediate_size, config.embedding_size)
        self.ffn_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = LayerNorm(config, eps=1e-12)

    def forward(self, hidden_states, encoder_output, output_mask, mutual_mask):
        # self attention
        attention_output = self.self_attention((hidden_states, hidden_states, hidden_states), output_mask)
        dense_output = self.selfatt_addnorm_dense(attention_output)
        drop_output = self.selfatt_addnorm_dropout(dense_output)
        final_attention_output = self.selfatt_addnorm_layernorm(drop_output+hidden_states)

        # attention to the encoder
        attention_output = self.multual_attention((final_attention_output, encoder_output, encoder_output), mutual_mask)
        dense_output = self.multualatt_addnorm_dense(attention_output)
        drop_output = self.multualatt_addnorm_dropout(dense_output)
        final_attention_output = self.multualatt_addnorm_layernorm(drop_output+final_attention_output)

        # position-wise
        ff_output = self.ffn_dropout(self.ffn_dense2(F.relu(self.ffn_dense1(final_attention_output))))
        hidden_states = self.layernorm(ff_output+final_attention_output)
        
        return hidden_states


class Decoder(nn.Module):
    """ Decoder """
    def __init__(self, config):
        super(Decoder, self).__init__()
        layer = DecoderLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, encoder_output, output_mask, mutual_mask):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, encoder_output, output_mask, mutual_mask)
        return hidden_states


class ForwardTransformer(nn.Module):
    def __init__(self, config):
        super(ForwardTransformer, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.predict_layer = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.num_attention_heads = config.num_attention_heads

    def forward(self, input_ids, output_ids, input_mask, output_mask, mutual_mask):
        input_embedding = self.embeddings(input_ids)
        output_embedding = self.embeddings(output_ids)
        input_mask = get_attention_mask(input_mask, self.num_attention_heads)
        output_mask = get_attention_mask(output_mask, self.num_attention_heads)
        mutual_mask = get_attention_mask(mutual_mask, self.num_attention_heads)
        encoder_output = self.encoder(input_embedding, input_mask)
        decoder_output = self.decoder(output_embedding, encoder_output, output_mask, mutual_mask)
        predict_output = self.predict_layer(decoder_output)
        return predict_output
