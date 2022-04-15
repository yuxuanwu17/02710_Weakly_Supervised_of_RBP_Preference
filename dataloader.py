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


def split_training_validation(classes, validation_size=0.2, shuffle=False):
    """split sampels based on balnace classes"""
    num_samples = len(classes)
    classes = np.array(classes)
    classes_unique = np.unique(classes)
    num_classes = len(classes_unique)
    indices = np.arange(num_samples)
    # indices_folds=np.zeros([num_samples],dtype=int)
    training_indice = []
    training_label = []
    validation_indice = []
    validation_label = []
    for cl in classes_unique:
        indices_cl = indices[classes == cl]
        num_samples_cl = len(indices_cl)

        # split this class into k parts
        if shuffle:
            random.shuffle(indices_cl)  # in-place shuffle

        # module and residual
        num_samples_each_split = int(num_samples_cl * validation_size)
        res = num_samples_cl - num_samples_each_split

        training_indice = training_indice + [val for val in indices_cl[num_samples_each_split:]]
        training_label = training_label + [cl] * res

        validation_indice = validation_indice + [val for val in indices_cl[:num_samples_each_split]]
        validation_label = validation_label + [cl] * num_samples_each_split

    training_index = np.arange(len(training_label))
    random.shuffle(training_index)
    training_indice = np.array(training_indice)[training_index]
    training_label = np.array(training_label)[training_index]

    validation_index = np.arange(len(validation_label))
    random.shuffle(validation_index)
    validation_indice = np.array(validation_indice)[validation_index]
    validation_label = np.array(validation_label)[validation_index]

    return training_indice, training_label, validation_indice, validation_label


if __name__ == '__main__':
    # unit test
    # data_file = "/Users/yuxuan/Desktop/iDeepS-master/datasets/clip/10_PARCLIP_ELAVL1A_hg19/30000/training_sample_0/sequences.fa"
    # data = load_data_file(data_file)
    # cls_name = read_class_name(data_file)

    # with open("data/egi_values.npy", 'rb') as f:
    #     egi_val = np.load(f, allow_pickle=True)
    # with open("data/seq_demo.npy", 'wb') as f:
    #     np.save(f, data)
    # with open("data/class_demo.npy", 'wb') as f:
    #     np.save(f, cls_name)

    with open("data/seq_demo.npy", 'rb') as f:
        seq = np.load(f, allow_pickle=True)
    with open("data/class_demo.npy", 'rb') as f:
        cls_name = np.load(f, allow_pickle=True)
    print(seq)
    print(cls_name)
