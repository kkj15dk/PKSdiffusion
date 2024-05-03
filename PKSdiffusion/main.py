import torch
from denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from utils import *
from Bio import SeqIO

set_seed(42) # set the random seed

model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 21
)

print("Model parameters: ", count_parameters(model))

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 3592,
    # seq_length = 40,
    timesteps = 1000,
    objective = 'pred_noise',
)

test = False
alignment = False

# aa_file = "clustalo_alignment.aln"
aa_file = "PKSs.fa"
if not test:
    if not alignment:
        train_record_aa = [record for record in SeqIO.parse(aa_file, "fasta")]
        seqs = [str(record.seq) for record in train_record_aa] # SOME OF THESE SEQS HAVE X AS A CHARACTER
        invalid_seqs = [seq for seq in seqs if "X" in seq]
        print("There are " + str(len(invalid_seqs)) + " sequences with X as a character. These will be removed from the training set.")
        seqs = [seq for seq in seqs if "X" not in seq]
        seqs = list(set(seqs)) # remove duplicates
        random.shuffle(seqs) # shuffle sequences
        characters = "ACDEFGHIKLMNPQRSTVWY-"
        max_len = max([len(seq) for seq in seqs])
        seqs = [pad_string(seq, max_len) for seq in seqs] # pad sequences to max length
    elif alignment:
        train_record_aa = [record for record in SeqIO.parse(aa_file, "fasta")]
        seqs = [str(record.seq) for record in train_record_aa] # SOME OF THESE SEQS HAVE X AS A CHARACTER
        invalid_seqs = [seq for seq in seqs if "X" in seq]
        print("There are " + str(len(invalid_seqs)) + " sequences with X as a character.")
        seqs = [seq for seq in seqs if "X" not in seq]
        seqs = list(set(seqs)) # remove duplicates
        random.shuffle(seqs) # shuffle sequences
        characters = "ACDEFGHIKLMNPQRSTVWY-"
        max_len = max([len(seq) for seq in seqs])
    print("THIS IS NOT A TEST. I REPEAT, THIS IS NOT A TEST:")
    print("Max length sequence is: " + str(max_len))

# training_seq = torch.rand(64, 32, 128) # features are normalized from 0 to 1
# dataset = Dataset1D(training_seq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

if test:
    print("This is a test using 1000 random sequences with predefined rules.")
    if alignment:
        seqs = random_aa_seq(1000)
        characters = "ACDEFGHIKLMNPQRSTVWY-"
        max_len = max([len(seq) for seq in seqs])
    elif not alignment:
        seqs = random_aa_seq_unaligned(1000)
        characters = "ACDEFGHIKLMNPQRSTVWY-" # "ACDEFGHIKLMNPQRSTVWY-<>"
        # seqs = [ ">" + seq + "<" for seq in seqs]
        max_len = max([len(seq) for seq in seqs])
        seqs = [pad_string(seq, max_len) for seq in seqs]
    print("Test of one_hot_encode and one_hot_decode:")
    print("Max length sequence is: " + str(max_len))
    print(seqs[0])
    aa_OHE = one_hot_encode(seqs[0], characters=characters)
    print(one_hot_decode(aa_OHE, characters=characters))

# Create a Dataset
# training_seq = torch.stack([one_hot_encode(seq, characters) for seq in seqs])
# dataset = Dataset1D(training_seq)
dataset = MyIterDataset(OHEAAgen, seqs, len(seqs), characters)

# loss = diffusion(training_seq)
# loss.backward()

# Or using trainer

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 100000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    results_folder="./resultsUNET"
)
trainer.load("50")
trainer.train()

# after a lot of training

sampled_seq = diffusion.sample(batch_size = 10)
print(sampled_seq.shape)
for i in range(sampled_seq.shape[0]):
    print(one_hot_decode(sampled_seq[i], characters=characters))
