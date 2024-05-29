import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
import random
import numpy as np
import matplotlib.pyplot as plt

def quick_loss_plot(train_data, label, filename, loss_type="Loss"):
    '''
    Plot the loss trajectory for each train/test data.

    Parameters:
    - data_label_list (list): A list of tuples containing train_data, test_data, and label.
    - loss_type (str): The type of loss to be plotted. Default is "BCEWithLogitsLoss".

    Returns:
    - None

    This function plots the loss trajectory for each train/test data in the data_label_list.
    It uses matplotlib to create a line plot with train_data plotted with dashed lines and test_data plotted with solid lines.
    The label parameter is used to provide a label for each line in the plot.
    The plot is saved as "loss_plot.png" in the current directory.
    The current figure is then cleared.

    Example usage:
    >>> data_label_list = [(train_data1, "Label1"), (train_data2, "Label2")]
    >>> quick_loss_plot(data_label_list, loss_type="MSELoss")
    '''

    plt.clf()
    plt.plot(train_data, color="C1", label=f"{label} Train", linewidth=3.0)

    plt.legend()
    plt.ylabel(loss_type)
    plt.xlabel("Steps")

    plt.legend(bbox_to_anchor=(0.75, 1), loc='upper left')
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)

    plt.savefig(filename + '.png')
    plt.clf()  # clear the current figure

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def one_hot_encode(seq, characters = "ACDEFGHIKLMNPQRSTVWY-", max_len = 40):
    """
    Given an AA sequence and a string of characters, return its one-hot encoding based on the characters provided.
    """
    # Make sure seq has only allowed bases, and the padding character. Holy fuck im making this convoluted, whoops
    allowed = set(characters + "-")

    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(f"Sequence contains chars not in allowed alphabet ({characters}): {invalid}")
        
    # Dictionary returning class indices for each nucleotide or amino acid

    dictionary = {x: i for i, x in enumerate(characters)}
    
    # Create array from nucleotide sequence
    tensor = F.one_hot(torch.tensor([dictionary[x] for x in seq]), num_classes=len(characters))

    # Calculate padding size
    pad_len = max_len - tensor.size(0)
    # Pad tensor randomly on right or left side
    pad = torch.zeros((pad_len, len(characters)))
    if random.choice([True, False]):  # Randomly choose True or False
        tensor = torch.cat((pad, tensor))
    else:
        tensor = torch.cat((tensor, pad))

    tensor = tensor.permute(1, 0)

    return tensor

def one_hot_decode(tensor, characters = "ACDEFGHIKLMNPQRSTVWY-", cutoff = 0.3):
    """
    Given a one-hot encoded tensor and a string of characters, return the decoded sequence based on the characters provided.
    """
    # Dictionary returning nucleotides or amino acids for each class index
    dictionary = {i: x for i, x in enumerate(characters)}

    # Create sequence from class indices
    seq = []
    for i in range(tensor.shape[1]):
        if torch.all(tensor[:, i] < cutoff):
            seq.append('-')
        else:
            seq.append(dictionary[torch.argmax(tensor[:, i]).item()])

    return "".join(seq)

class MyIterDataset(IterableDataset):
    def __init__(self, generator_function, seqs, len, characters="ACDEFGHIKLMNPQRSTVWY-", max_len = 40):
        self.generator_function = generator_function
        self.seqs = seqs
        self.len = len
        self.characters = characters
        self.length = max_len

    def __iter__(self):
        # Create a generator object
        generator = self.generator_function(self.seqs, self.characters, self.length)
        for item in generator:
            yield item.float()
    
    def __len__(self):
        return self.len

def OHEAAgen(seqs, characters="ACDEFGHIKLMNPQRSTVWY-", length=40):
    # yield from record_gen
    for seq in seqs:
        # seq = pad_string(seq, length=3592)
        # seq = pad_string(seq, length=1800)
        # seq = pad_string(seq, length=length)
        # seq = pad_string(seq, length=3592)
        seq = one_hot_encode(seq, characters, length)
        # seq = 2*seq - 1 # Go from 0:1 to -1:1

        yield seq

# Description: Generate random amino acid sequences for testing
def random_aa_seq(n):
    lsseq = []
    for i in range(n):
        # Generate a random aa sequence
        seq = "M"
        for j in range(3):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY-")
        seq += "HINQA"
        seq += random.choice(["----","ACDE"])
        for j in range(2):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY-")
        seq += random.choice(["----","FGHI"])
        for j in range(2):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY-")
        seq += random.choice(["----","KLMN"])
        for j in range(3):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY-")
        seq += random.choice(["----", "PQRS"])
        for j in range(4):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY-")
        seq += random.choice(["----", "TVWY"])
        # Print the sequence
        # print(seq)
        lsseq.append(seq)
    return lsseq

def random_aa_seq_unaligned(n):
    lsseq = []
    for i in range(n):
        # Generate a random aa sequence
        seq = "M"
        for j in range(3):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY")
        seq += "HINQA"
        seq += random.choice(["","ACDE"])
        for j in range(2):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY")
        seq += random.choice(["","FGHI"])
        for j in range(2):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY")
        seq += random.choice(["","KLMN"])
        for j in range(3):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY")
        seq += random.choice(["", "PQRS"])
        for j in range(4):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY")
        seq += random.choice(["", "TVWY"])
        # Print the sequence
        # print(seq)
        lsseq.append(seq)
    return lsseq

def pad_string(string, length, padding_value='-'):
    """
    Pads or truncates a string to a specified length with a given padding value.

    Args:
        string (str): The input string to be padded.
        length (int): The desired length of the string after padding.
        padding_value (str, optional): The character used for padding. Defaults to '-'.

    Returns:
        str: The padded string.
    """
    # Pad Left or right, by random choice
    if len(string) < length:
        pad_len = length - len(string)
        if random.choice([True, False]):  # Randomly choose True or False
            string = padding_value * pad_len + string  # Pad left
        else:
            string = string + padding_value * pad_len  # Pad right
    else:
        string = string[:length]
    return string

    # # Pad randomly to length
    # if len(string) < length:
    #     rand_len = random.randint(len(string), length)
    #     left_pad = rand_len - len(string)
    #     right_pad = length - rand_len
    #     string = padding_value * left_pad + string + padding_value * right_pad
    # else:
    #     string = string[:length]
    # return string

    # # Pad right
    # if len(string) < length:
    #     right_pad = length - len(string)
    #     string = string + padding_value * right_pad
    # else:
    #     string = string[:length]
    # return string

# Set a random seed in a bunch of different places
def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The random seed to set.

    Returns:
        None
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set as {seed}")

def CVAEcollate_fn(batch):
    # stack the inputs into a single tensor
    inputs = torch.stack(batch)
    return inputs