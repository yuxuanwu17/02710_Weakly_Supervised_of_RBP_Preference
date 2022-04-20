import numpy as np
import os
import gzip
import random
import pdb


def load_data_file(inputfile):
    """
        Load data matrices from the specified folder.
    """
    seq_info = read_seq(inputfile)
    return seq_info


def read_seq(seq_file):
    seq_list = []
    seq = ''
    with open(seq_file, 'r') as f:
        for line in f:
            if line[0] == '>':
                if len(seq):
                    seq_array = seq
                    seq_list.append(seq_array)
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq):
            seq_array = seq
            seq_list.append(seq_array)

    return np.array(seq_list)


def read_class_name(seq_file):
    cls_name_list = []
    with open(seq_file, 'r') as f:
        for line in f:
            if line[0] == ">":
                cls_name = line[-2]
                cls_name_list.append(cls_name)
    return cls_name_list


def embed(sequence, instance_len, instance_stride):
    instance_num = int((len(sequence) - instance_len) / instance_stride) + 1
    bag = []
    for i in range(instance_num):
        instance = sequence[i * instance_stride:i * instance_stride + instance_len]
        instance = one_hot_encode(instance)
        bag.append(instance)
    bag = np.stack(bag).astype(np.float64)
    return bag


def one_hot_encode(seq):
    mapping = dict(zip("ACGT", range(4)))
    seq2 = [mapping[i] for i in seq ]
    return np.eye(4)[seq2]


def create_bag(train_seq, valid_seq, instance_len=40, instance_stride=5):
    # length and strides could be simulated via cross validation

    train_bags = []
    for seq in train_seq:
        ont_hot_bag = embed(seq, instance_len, instance_stride)
        train_bags.append(ont_hot_bag)

    train_bags = np.asarray(train_bags)

    valid_bags = []
    for seq in valid_seq:
        ont_hot_bag = embed(seq, instance_len, instance_stride)
        valid_bags.append(ont_hot_bag)

    valid_bags = np.asarray(valid_bags)

    return train_bags, valid_bags


def train_test_split(seq, cls_name, ratio=0.8):
    assert len(seq) == len(cls_name)
    train_num = int(len(seq) * ratio)
    temp = list(zip(seq, cls_name))
    random.shuffle(temp)
    seq, cls_name = zip(*temp)

    train_seq = np.asarray(seq)[:train_num]
    train_cls = np.asarray(cls_name, dtype=float)[:train_num]

    val_seq = np.asarray(seq)[train_num:]
    val_cls = np.asarray(cls_name, dtype=float)[train_num:]

    return train_seq, val_seq, train_cls, val_cls


def str2token(seq):
    seq_dict = {'A': "0", 'C': "1", 'G': "2", 'T': "3", 'N': "4"}
    token = ""
    for i in seq:
        token += seq_dict[i]
    return token


if __name__ == '__main__':
    # unit test
    # data_file = "/Users/yuxuan/Desktop/iDeepS-master/datasets/clip/10_PARCLIP_ELAVL1A_hg19/30000/training_sample_0/sequences.fa"
    # data = load_data_file(data_file)
    # cls_name = read_class_name(data_file)

    # with open("data/seq_demo.npy", 'wb') as f:
    #     np.save(f, data)
    # with open("data/class_demo.npy", 'wb') as f:
    #     np.save(f, cls_name)

    # print(one_hot_encode("AGCTCCTTCT"))
    #
    # seq = np.load("data/seq_demo.npy")
    # cls_name = np.load("data/class_demo.npy")

    # seq = [str2token(i) for i in seq]

    # train_seq, val_seq, train_cls, val_cls = train_test_split(seq, cls_name)
    #
    # train_bags, valid_bags = create_bag(train_seq, val_seq)

    # with open("data/train_bags_demo.npy", 'wb') as f:
    #     np.save(f, train_bags)
    # with open("data/train_clsname_demo.npy", 'wb') as f:
    #     np.save(f, train_cls)
    # with open("data/valid_bags_demo.npy", 'wb') as f:
    #     np.save(f, valid_bags)
    # with open("data/valid_clsname_demo.npy", 'wb') as f:
    #     np.save(f, val_cls)

    valid_bags = np.load("data/valid_bags_demo.npy", allow_pickle=True)
    train_bags = np.load("data/train_bags_demo.npy", allow_pickle=True)
    #
    # print(train_bags.shape)
    # print(valid_bags.shape)

    # print(train_bags[0].shape) # 13 =(101-40)/5 = 12+1 , 40 instance length , # AGCTN # 4
    # print(train_bags[0][0]) # 13 =(101-40)/5 = 12+1 , 40 instance length , # AGCTN # 4
