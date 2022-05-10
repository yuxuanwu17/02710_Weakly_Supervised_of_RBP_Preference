import pandas as pd
import numpy as np
import gzip

import torch
from torch.utils.data import DataLoader, Dataset

def read_file_helper(file_path):
    res = []
    with gzip.open(file_path, "rb") as f:
        for line in f.readlines():
            line = str(line)[2:-3]
            
            if line.startswith(">"):
                indicator = 0
                tmp = []
                loc, y = line.strip().split(";")
                chr_num, sign, start, end = loc[2:].split(",")
                tmp.extend([chr_num, sign, int(start), int(end), y[-1]])
            
            else:
                indicator += 1
                tmp.append(line)
                if indicator == 2:
                    res.append(tmp)
    df = pd.DataFrame(res, columns = ["chr_num", "sign", "start", "end", "y", "seq_part1", "seq_part2"])
    df["seq"] = df["seq_part1"] + df["seq_part2"]
    df["y"] = df["y"].astype(int)
    return df

def read_structure_file(file_path):
    res = []
    with open(file_path, 'r') as f:
        for index, line in enumerate(f.readlines()):
            if index%8 == 1:
                res.append(line.strip().lower())
    return res
            

def embed(sequence, instance_len, instance_stride, one_hot_encode):
    instance_num = int((len(sequence) - instance_len) / instance_stride) + 1
    bag = []
    for i in range(instance_num):
        instance = sequence[i * instance_stride:i * instance_stride + instance_len]
        instance = one_hot_encode(instance)
        bag.append(instance)
    bag = np.stack(bag).astype(float)
    return bag

def one_hot_encode(seq):
    arrays = [np.array([1, 0, 0, 0]),
             np.array([0, 1, 0, 0]),
             np.array([0, 0, 1, 0]),
             np.array([0, 0, 0, 1]),
             np.array([0.25, 0.25, 0.25, 0.25])]
             
    mapping = dict(zip("ACGTN", arrays))
   
    return np.vstack([mapping[i] for i in seq])

def one_hot_encode_rna_structure(seq):
    arrays = [np.array([1, 0, 0, 0, 0, 0]),
              np.array([0, 1, 0, 0, 0, 0]),
              np.array([0, 0, 1, 0, 0, 0]),
              np.array([0, 0, 0, 1, 0, 0]),
              np.array([0, 0, 0, 0, 1, 0]),
              np.array([0, 0, 0, 0, 0, 1])]
    
    mapping = dict(zip("shitmf", arrays))
   
    return np.vstack([mapping[i] for i in seq])
              

def create_bag(seqs, one_hot_encode_method, instance_len=40, instance_stride=5):
    bags = []
    for seq in seqs:
        bags.append(embed(seq, instance_len, instance_stride, one_hot_encode_method)) 
        
    return np.array(bags)

class LibriSamples(Dataset):
    def __init__(self, data_path):
        df = read_file_helper(data_path)
        df["seq"] = df["seq_part1"] + df["seq_part2"]
        self.X, self.Y = create_bag(df["seq"], one_hot_encode).astype(float), df["y"].to_numpy()
        
        assert len(self.X) == len(self.Y)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        y = [0, 1] if self.Y[item] == 1 else [1, 0]
        return self.X[item], np.array(y)
    
class LibriSamplesWithStructure(Dataset):
    def __init__(self, data_path, structure_path):
        df = read_file_helper(data_path)
        df["seq"] = df["seq_part1"] + df["seq_part2"]
        self.X, self.Y = create_bag(df["seq"], one_hot_encode), df["y"].to_numpy()
        assert len(self.X) == len(self.Y)
        
        structures = read_structure_file(structure_path)
        # print(structures[:10])
        self.structures = create_bag(structures, one_hot_encode_rna_structure)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        y = [0, 1] if self.Y[item] == 1 else [1, 0]
        return self.X[item], np.array(y), self.structures[item]
    

if __name__ == "__main__":
    batch_size = 1
    path = '../iDeepS/datasets/clip/10_PARCLIP_ELAVL1A_hg19'
    train_data_path = path + "/30000/training_sample_0/sequences.fa.gz"
    valid_data_path = path + "/30000/test_sample_0/sequences.fa.gz"
    train_data = LibriSamples(train_data_path)
    valid_data = LibriSamples(valid_data_path)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size, shuffle=True)
    
    train_structure_path = path + "/30000/training_sample_0/sequence_structures_forgi.out"
    validate_structure_path = path + "/30000/test_sample_0/sequence_structures_forgi.out"
    train_data = LibriSamplesWithStructure(train_data_path, train_structure_path)
    valid_data = LibriSamplesWithStructure(valid_data_path, validate_structure_path)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size, shuffle=True)