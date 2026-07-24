# -*-coding:utf-8-*-

import numpy as np
import random

#########---- pad ----#########
def pad(input_arr, target_length, val=0, dim=1):
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    npad[dim] = (0, target_length - shp[dim])
    return np.pad(input_arr, pad_width=npad, mode='constant', constant_values=val)

def pad_sequence(seq, target_length, method='random', indices=None):
    """
    Pads a sequence to the target length using specified method.

    Parameters:
    seq (list or np.array): The original sequence to pad.
    target_length (int): The desired length after padding.
    method (str): Method to use for padding ('random' or 'sequential').

    Returns:
    np.array: Padded sequence.
    """
    current_length = len(seq)
    if current_length >= target_length:
        return np.array(seq[:target_length]), indices

    # Determine the padding length
    padding_length = target_length - current_length

    # Choose padding values
    if method == 'random':
        if indices is None:
            indices = [random.randint(0, current_length - 1) for _ in range(padding_length)]
        padding_values = np.array([seq[idx] for idx in indices])
    elif method == 'sequential':
        padding_values = seq[:padding_length]
        indices = np.arange(padding_length)
    else:
        raise ValueError("Method must be either 'random' or 'sequential'")

    # Concatenate the original sequence with the padding values
    padded_seq = np.concatenate((seq, padding_values), 0)
    return padded_seq, indices


def pad_ne(seq, target_length, indices):

    current_length = len(seq)
    if current_length >= target_length:
        return np.array(seq[:target_length])

    # Choose padding values
    add = np.arange(current_length, target_length).reshape(-1, 1)
    seq1 = np.concatenate((seq[:, 0].reshape(-1, 1), add.reshape(-1, 1)))

    padding_values = [seq[idx, 1:] for idx in indices]
    concatenated_array = np.array(padding_values)

    seq2 = np.concatenate((seq[:, 1:].reshape(-1, seq.shape[1]-1), concatenated_array))

    # Concatenate the original sequence with the padding values
    padded_seq = np.concatenate((seq1, seq2), 1)

    return padded_seq

