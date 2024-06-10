import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
import random
import numpy as np
import matplotlib.pyplot as plt
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import json
import os

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

def plot_dataset(dataset, bins = None):
    # Create a dictionary to hold datasets divided by label
    divided_dataset = {}

    # Iterate over the dataset
    for seq, cl in dataset:
        cl = cl.item()
        # If the label is not in the dictionary, add it with an empty list
        if cl not in divided_dataset:
            divided_dataset[cl] = []
        # Append the sequence to the appropriate list
        divided_dataset[cl].append(seq.shape[-1])

    # Iterate over the divided dataset
    for label, data in divided_dataset.items():
        mean = sum(data) / len(data)
        plt.hist(data, bins=bins, label=str(label), alpha=1)
        plt.legend()
        plt.xlabel("seq length",fontsize=14)
        plt.ylabel("count",fontsize=14)
        plt.xlim(0, 2000)
        # plt.ylim(0, 32)
        plt.savefig(f'train_hist_{label}.png')
        plt.title(f"Distribution for label {label} with mean {mean:.2f} and std {np.std(data):.2f}")
        plt.savefig(f"train_hist_{label}.png")
        plt.clf()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def one_hot_encode(seq, characters = "ACDEFGHIKLMNPQRSTVWY-", max_len = 40):
    """
    Given an AA sequence and a string of characters, return its one-hot encoding based on the characters provided.
    """

    allowed = set(characters)

    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(f"Sequence contains chars not in allowed alphabet ({characters}): {invalid}")
        
    # Dictionary returning class indices for each nucleotide or amino acid

    dictionary = {x: i for i, x in enumerate(characters)}
    
    # Create array from nucleotide sequence
    tensor = F.one_hot(torch.tensor([dictionary[x] for x in seq]), num_classes=len(characters))

    tensor = tensor.permute(1, 0)

    return tensor

def one_hot_decode(tensor, characters = "ACDEFGHIKLMNPQRSTVWY-", cutoff = 0.5): # cutoff = 0.5 means all classes are negative in the model output before normalisation from -1:1 to 0:1. This will be interpreted as padding.
    """
    Given a one-hot encoded tensor and a string of characters, return the decoded sequence based on the characters provided.
    """
    # If the tensor is not a batch, add an extra dimension to make it a batch of size 1
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)
    
    # Dictionary returning nucleotides or amino acids for each class index
    dictionary = {i: x for i, x in enumerate(characters)}

    # Create sequence from class indices
    # Find the indices of the maximum values along the last dimension
    indices = torch.argmax(tensor, dim=-2)

    # Create a mask where all values in the tensor are less than the cutoff
    mask = torch.all(tensor < cutoff, dim=-2)

    # Replace the indices in the mask with the length of the characters (which is not a valid index)
    indices = torch.where(mask, torch.tensor(len(characters), device=tensor.device), indices)

    # Convert the tensor of indices to a list of indices
    indices = indices.tolist()

    # Use the dictionary to map the indices to characters
    seq = [[dictionary.get(idx, 'X') for idx in batch] for batch in indices]
    # seq = [[dictionary.get(idx, '-') for idx in batch] for batch in indices]

    # Join the characters to form the sequences
    seq = ["".join(batch) for batch in seq]

    return seq

class MyIterDataset(IterableDataset):
    def __init__(self, generator_function, seqs, len, characters="ACDEFGHIKLMNPQRSTVWY-", max_len = 40, varying_length = False, varying_length_resolution = 4):
        self.generator_function = generator_function
        self.seqs = seqs
        self.len = len
        self.characters = characters
        self.max_len = max_len
        self.varying_length = varying_length
        self.varying_length_resolution = varying_length_resolution

    def __iter__(self):
        # Create a generator object
        generator = self.generator_function(self.seqs, self.characters, self.max_len, varying_length=self.varying_length, varying_length_resolution=self.varying_length_resolution)
        for seq, cl in generator:
            yield seq, cl

    def __len__(self):
        return self.len

def collate_fn(batch):
    # Sort the batch in the descending order
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    # Separate sequence and target
    sequences, cls = zip(*sorted_batch)
    length = max([seq.shape[1] for seq in sequences])
    
    sequences_padded = []
    masks = []

    for seq in sequences:
        seq, mask = get_mask(seq, length)
        sequences_padded.append(seq)
        masks.append(mask)
    
    sequences_padded = torch.stack(sequences_padded)
    mask = torch.stack(masks)
    cls = torch.stack(cls)

    return sequences_padded, mask, cls

def OHEAAgen(seqs, characters="ACDEFGHIKLMNPQRSTVWY-", length=40, varying_length=False, varying_length_resolution = 4):
    # yield from record_gen
    # First adds '-' to the string until length is divisible by varying_length_resolution, then onehotencodes it, then pads it to length, then creates a mask for the padding
    for seq in seqs:
        seq, cl = seq
        cl = torch.tensor(cl)

        seq = pad_string(seq, length = length, varying_length = varying_length, varying_length_resolution = varying_length_resolution) # varying_length = True, pad to len % 4 == 0
        seq = one_hot_encode(seq, characters, length)

        yield seq.float(), cl

def get_mask(seq, length):
    mask = torch.ones(length)
    mask[seq.size(1):] = 0
    seq = torch.cat((seq, torch.zeros((seq.size(0), length - seq.size(1)))), dim=1)
    return seq, mask

# Description: Generate random amino acid sequences for testing
def random_aa_seq(n):
    lsseq = []
    for i in range(n):
        cl = 1 # class label
        # Generate a random aa sequence
        seq = "M"
        for j in range(3):
            seq += random.choice(list("ACDEFGHIKLMNPQRSTVWY") + [""])
        seq += "HINQ"
        if random.choice([True, False]):
            seq += "DTFG"
        else:
            seq += "ACDE"
            cl = 2
        for j in range(2):
            seq += random.choice(list("ACDEFGHIKLMNPQRSTVWY") + [""])
        seq += random.choice(["","FGH"])
        for j in range(2):
            seq += random.choice(list("ACDEFGHIKLMNPQRSTVWY") + [""])
        seq += random.choice(["","IKLMN"])
        for j in range(3):
            seq += random.choice(list("ACDEFGHIKLMNPQRSTVWY") + [""])
        seq += random.choice(["", "PQRS"])
        for j in range(4):
            seq += random.choice(list("ACDEFGHIKLMNPQRSTVWY") + [""])
        seq += random.choice(["", "TVWY"])
        seq += random.choice(["", "MEG"])
        seq += random.choice(["", "VWY"])
        seq += random.choice(["", "END"])
        # Print the sequence
        # print(seq)
        lsseq.append((seq, cl))
    return lsseq

def write_fasta(seqs, filename):
    records = []
    if os.path.exists('test.fa'):
        return
    for i, (seq, cl) in enumerate(seqs):
        record = SeqRecord(Seq(seq), id=f"{i}|{cl}", description="")
        records.append(record)
    SeqIO.write(records, filename, "fasta")

def load_fasta(aa_file = "NRPSs_mid-1800.fa", label_file = 'labels.json', characters = "ACDEFGHIKLMNPQRSTVWY-", varying_length=False, varying_length_resolution = 4):
    label_dict = json.loads(open(label_file).read())
    train_record_aa = [record for record in SeqIO.parse(aa_file, "fasta")]
    seqs = []
    for record in train_record_aa:
        description = record.description.split('|')[-1]
        if description in label_dict:
            seq = str(record.seq)
            cl = label_dict[description]['class']
            seqs.append((seq, cl))
    print("There are " + str(len(seqs)) + " sequences in the dataset with correct labeling.")
    seqs = [seq for seq in seqs if set(seq[0]).issubset(characters)]
    print("There are " + str(len(seqs)) + " sequences when removing unimplemented amino acids.")
    seqs = list(set(seqs)) # remove duplicates
    print("There are " + str(len(seqs)) + " sequences when removing duplicates. This is the final dataset.")
    random.shuffle(seqs) # shuffle sequences
    max_len = max([len(seq[0]) for seq in seqs])
    min_len = min([len(seq[0]) for seq in seqs])
    print("Max length sequence is: " + str(max_len))
    print("Min length sequence is: " + str(min_len))
    if varying_length:
        max_len = ((max_len + varying_length_resolution - 1) // varying_length_resolution) * varying_length_resolution
        print("Max length sequence is: " + str(max_len) + " after padding to length divisible by " + str(varying_length_resolution) + ".")

    return seqs, max_len

def pad_string(string, length, padding_value='-', varying_length = False, varying_length_resolution = 4):
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
    if varying_length:
        pad_len = -len(string) % varying_length_resolution # Two downsamplings in U-Net give 2*2, seqlength should be divisible by 4
        # if random.choice([True, False]):  # Randomly choose True or False
        #     string = padding_value * pad_len + string  # Pad left
        # else:
        #     string = string + padding_value * pad_len  # Pad right
        string = string + padding_value * pad_len  # Pad right
    elif len(string) < length:
        pad_len = length - len(string)
        if random.choice([True, False]):  # Randomly choose True or False
            string = padding_value * pad_len + string  # Pad left
        else:
            string = string + padding_value * pad_len  # Pad right
    else:
        string = string[:length]
    return string

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

if __name__ == '__main__':
    pass
    # print