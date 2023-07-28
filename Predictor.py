# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port 3303 Predictor.py --predict_file /home/ubuntu/Experimental_Data/v1_5UTR_seqs_with_v1Label.fasta --outdir /home/ubuntu/Experimental_Data/try --outfilename try_RVACv1


import os
from Bio import SeqIO
import sys

# import argparse
# from argparse import Namespace
# import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

# import esm
# from esm.data import *
# from esm.model.esm2_secondarystructure import ESM2 as ESM2_SISS
from esm.model.esm2 import ESM2 as ESM2_SISS
# from esm.model.esm2_supervised import ESM2
from esm import Alphabet, FastaBatchedDataset#, ProteinBertModel, pretrained, MSATransformer


import numpy as np
import pandas as pd
import random
# import math
# import scipy.stats as stats
# from scipy.stats import spearmanr, pearsonr
# from sklearn import preprocessing
# from copy import deepcopy
from tqdm import tqdm#, trange
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import KFold
# from torch.optim.lr_scheduler import StepLR
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel
# from torch.utils.data.distributed import DistributedSampler
from io import StringIO

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

# parser = argparse.ArgumentParser()
# parser.add_argument('--device_ids', type=str, default='0', help="Training Devices")
# parser.add_argument('--local-rank', type=int, default=-1, help="DDP parameter, do not modify")

# parser.add_argument('--outdir', type=str, default = '/home/ubuntu/Experimental_Data/try')
# parser.add_argument('--outfilename', type=str, default = 'try_RVACv1')
# parser.add_argument('--predict_file', type = str, default = '/home/ubuntu/Experimental_Data/v1_5UTR_seqs_with_v1Label.fasta')
# args = parser.parse_args()
# print(args)

global modelfile, layers, heads, embed_dim, batch_toks, inp_len, device_ids, device
modelfile = 'model.pkl'

# model_info = modelfile.split('/')[-1].split('_')
# for item in model_info:
#     if 'layers' in item: 
#         layers = int(item[0])
#     elif 'heads' in item:
#         heads = int(item[:-5])
#     elif 'embedsize' in item:
#         embed_dim = int(item[:-9])
#     elif 'batchToks' in item:
#         batch_toks = 4096
        
layers = 6
heads = 16
embed_dim = 128
batch_toks = 4096

inp_len = 50
    
# device_ids = list(map(int, args.device_ids.split(',')))
# dist.init_process_group(backend='nccl')
# device = torch.device('cuda:{}'.format(device_ids[args.local_rank]))
device = "cpu"
# torch.cuda.set_device(device)

# local_rank = args.local_rank
local_rank = -1
# storage_id = int(device_ids[local_rank])
storage_id = 0

# repr_layers = [layers]
include = ["mean"]
    
class CNN_linear(nn.Module):
    def __init__(self, 
                 border_mode='same', filter_len=8, nbr_filters=120,
                 dropout1=0, dropout2=0):
        
        super(CNN_linear, self).__init__()
        
        self.embedding_size = embed_dim
        self.border_mode = border_mode
        self.inp_len = inp_len
        self.nodes = 40
        self.cnn_layers = 0
        self.filter_len = filter_len
        self.nbr_filters = nbr_filters
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = 0.5
        
        self.esm2 = ESM2_SISS(num_layers = layers,
                                 embed_dim = embed_dim,
                                 attention_heads = heads,
                                 alphabet = alphabet)
        
        self.conv1 = nn.Conv1d(in_channels = self.embedding_size, 
                      out_channels = self.nbr_filters, kernel_size = self.filter_len, padding = self.border_mode)
        self.conv2 = nn.Conv1d(in_channels = self.nbr_filters, 
                      out_channels = self.nbr_filters, kernel_size = self.filter_len, padding = self.border_mode)
        
        self.dropout1 = nn.Dropout(self.dropout1)
        self.dropout2 = nn.Dropout(self.dropout2)
        self.dropout3 = nn.Dropout(self.dropout3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features = embed_dim, out_features = self.nodes)
        self.linear = nn.Linear(in_features = self.nbr_filters, out_features = self.nodes)
        self.output = nn.Linear(in_features = self.nodes, out_features = 1)
        self.direct_output = nn.Linear(in_features = embed_dim, out_features = 1)
        self.magic_output = nn.Linear(in_features = 1, out_features = 1)
            
    def forward(self, tokens, need_head_weights=True, return_contacts=False, return_representation=True):
        
        # x = self.esm2(tokens, [layers], need_head_weights, return_contacts, return_representation)
        x = self.esm2(tokens, [layers])

        x = x["representations"][layers][:, 0]
        x_o = x.unsqueeze(2)
        
        x = self.flatten(x_o)
        o_linear = self.fc(x)
        o_relu = self.relu(o_linear)
        o_dropout = self.dropout3(o_relu)
        o = self.output(o_dropout)
        return o

def eval_step(dataloader, model, threshold = 0.5):
    model.eval()
    y_pred_list, y_prob_list = [], []
    ids_list, strs_list = [], []
    with torch.no_grad():
        # for (ids, strs, _, toks, _, _) in tqdm(dataloader):
        for ids, strs, toks in tqdm(dataloader):
            ids_list.extend(ids)
            strs_list.extend(strs)
            # toks = toks.to(device)
            
            # print(toks)
            logits = model(toks) 

            logits = logits.reshape(-1)
            y_prob = torch.sigmoid(logits)
            y_pred = (y_prob > threshold).long()
            
            
            y_prob_list.extend(y_prob.cpu().detach().tolist())
            y_pred_list.extend(y_pred.cpu().detach().tolist())
            
        data_pred = pd.DataFrame([ids_list, strs_list, y_prob_list, y_pred_list], index = ['ID', 'Sequence', "Probability as 5'UTR", "Prediction as 5'UTR"]).T
    return data_pred



def generate_dataset_dataloader(ids, seqs):
    # dataset = FastaBatchedDataset(ids, seqs, mask_prob = 0.0)
    dataset = FastaBatchedDataset(ids, seqs)
    batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=2)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            collate_fn=alphabet.get_batch_converter(), 
                                            batch_sampler=batches, 
                                            shuffle = False)
    print(f"{len(dataset)} sequences")
    return dataset, dataloader

def read_fasta(file):
    # 判断文件是否为空
    if os.path.getsize(file) == 0:
        print("Error: The file is empty!")
        sys.exit()

    ids = []
    sequences = []

    for record in SeqIO.parse(file, "fasta"):
        # 检查序列的开头是否为">"
#         if not record.id.startswith('>'):
#             print(f"Error: The sequence '{record.id}' is not properly formatted, it does not start with '>'. Skipping...")
#             continue

        # 检查序列是否只包含A, G, C, T
        sequence = str(record.seq).upper()[-inp_len:]
        if not set(sequence).issubset(set("AGCT")):
            print(f"Error: The sequence '{record.description}' contains invalid characters. Only A, G, C, T are allowed. Skipping...")
            continue

        # 将符合条件的序列添加到列表中
        ids.append(record.id)
        sequences.append(sequence)
    
    return ids, sequences

def read_raw(raw_input):
    ids = []
    sequences = []

    file = StringIO(raw_input)
    for record in SeqIO.parse(file, "fasta"):
        # 检查序列的开头是否为">"
#         if not record.id.startswith('>'):
#             print(f"Error: The sequence '{record.id}' is not properly formatted, it does not start with '>'. Skipping...")
#             continue

        # 检查序列是否只包含A, G, C, T
        sequence = str(record.seq).upper()[-inp_len:]
        if not set(sequence).issubset(set("AGCT")):
            print(f"Error: The sequence '{record.description}' contains invalid characters. Only A, G, C, T are allowed. Skipping...")
            continue

        # 将符合条件的序列添加到列表中
        ids.append(record.id)
        sequences.append(sequence)
    
    return ids, sequences

#######

# alphabet = Alphabet(mask_prob = 0.0, standard_toks = 'AGCT')
alphabet = Alphabet(prepend_toks=("<pad>", "<eos>", "<unk>"), standard_toks = 'AGCT', append_toks=("<cls>", "<mask>", "<sep>"))
# print(alphabet.tok_to_idx)
# assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}
alphabet.tok_to_idx = {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}

def predict_file(input_file):
    print('====Load Data====')
    ids, seqs = read_fasta(input_file)
    _, dataloader = generate_dataset_dataloader(ids, seqs)
        
    model = CNN_linear().to(device)
    # model.load_state_dict({k.replace('module.', ''):v for k,v in torch.load(modelfile, map_location=lambda storage, loc : storage.cuda(storage_id)).items()}, strict = False)
    model.load_state_dict({k.replace('module.', ''):v for k,v in torch.load(modelfile, map_location=torch.device('cpu')).items()}, strict = False)
    # model = DistributedDataParallel(model, device_ids=[device_ids[local_rank]], output_device=device_ids[local_rank], find_unused_parameters=True)

    print('====Predict====')
    pred = eval_step(dataloader, model)

    print(pred)
    # print('====Save Results====')         
    # if not os.path.exists(args.outdir): os.makedirs(args.outdir)
    # pred.to_csv(f'{args.outdir}/{args.outfilename}_prediction_results.csv', index = False)

def predict_raw(raw_input):
    print('====Parse Input====')
    ids, seqs = read_raw(raw_input)
    _, dataloader = generate_dataset_dataloader(ids, seqs)
        
    model = CNN_linear().to(device)
    # model.load_state_dict({k.replace('module.', ''):v for k,v in torch.load(modelfile, map_location=lambda storage, loc : storage.cuda(storage_id)).items()}, strict = False)
    model.load_state_dict({k.replace('module.', ''):v for k,v in torch.load(modelfile, map_location=torch.device('cpu')).items()}, strict = False)
    # model = DistributedDataParallel(model, device_ids=[device_ids[local_rank]], output_device=device_ids[local_rank], find_unused_parameters=True)

    print('====Predict====')
    pred = eval_step(dataloader, model)

    print(pred)