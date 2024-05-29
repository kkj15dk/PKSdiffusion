import torch
from denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from utils import *
from Bio import SeqIO

seed = 41
set_seed(seed) # set the random seed
print("seed set as " + str(seed))

model = Unet1D( # This UNET model cannot take in odd length inputs...
    dim = 64,
    # dim = 128,
    dim_mults = (1, 2, 4, 8),
    channels = 20,
    learned_sinusoidal_cond=True,
    random_fourier_features=True,
    # learned_variance=True, # Makes it crash
)

print("Model parameters: ", count_parameters(model))

test = False
alignment = False

# aa_file = "clustalo_alignment.aln"
# aa_file = "PKSs.fa"
aa_file = "NRPSs_mid-0-1800.fa"
if not test:
    if not alignment:
        train_record_aa = [record for record in SeqIO.parse(aa_file, "fasta")]
        characters = "ACDEFGHIKLMNPQRSTVWY"
        seqs = [str(record.seq) for record in train_record_aa] # SOME OF THESE SEQS HAVE UNIMPLEMENTED AA's AS A CHARACTERS
        print("There are " + str(len(seqs)) + " sequences in the daatset.")
        seqs = [seq for seq in seqs if set(seq).issubset(characters)]
        print("There are " + str(len(seqs)) + " sequences when removing unimplemented amino acids.")
        seqs = list(set(seqs)) # remove duplicates
        print("There are " + str(len(seqs)) + " sequences when removing duplicates. This is the final dataset.")
        random.shuffle(seqs) # shuffle sequences
        max_len = max([len(seq) for seq in seqs])
        # seqs = [pad_string(seq, max_len) for seq in seqs] # pad sequences to max length Should be done in dataloader/collate_fn
    elif alignment:
        train_record_aa = [record for record in SeqIO.parse(aa_file, "fasta")]
        seqs = [str(record.seq) + "---" for record in train_record_aa] # SOME OF THESE SEQS HAVE X AS A CHARACTER. Adding --- to pad for the length needed to the current UNET architecture, which is doesnt take all input lengths...
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
        characters = "ACDEFGHIKLMNPQRSTVWY" # "ACDEFGHIKLMNPQRSTVWY-<>"
        # seqs = [ ">" + seq + "<" for seq in seqs]
        max_len = max([len(seq) for seq in seqs])
        # seqs = [pad_string(seq, max_len) for seq in seqs]
    print("Test of one_hot_encode and one_hot_decode:")
    print("Max length sequence is: " + str(max_len))
    print(seqs[0])
    aa_OHE = one_hot_encode(seqs[0], characters=characters)
    print(one_hot_decode(aa_OHE, characters=characters))

diffusion = GaussianDiffusion1D(
    model,
    seq_length = max_len,
    # seq_length = 40,
    timesteps = 1000,
    # objective = 'pred_noise',
    # objective = 'pred_x0', 
    objective = 'pred_v',
    beta_schedule = 'cosine',
    # beta_schedule = 'linear',
    auto_normalize=True,
)

# Create a Dataset
# training_seq = torch.stack([one_hot_encode(seq, characters) for seq in seqs])
# dataset = Dataset1D(training_seq)
dataset = MyIterDataset(OHEAAgen, seqs, len(seqs), characters, max_len)

# loss = diffusion(training_seq)
# loss.backward()

# Or using trainer

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 1e-5,
    train_num_steps = 1000000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    save_and_sample_every = 10000,
    results_folder="./resultsTEST_NRPS_3",
)
# trainer.load("2")
diffusion.visualize_diffusion(next(iter(dataset)), [100*i for i in range(10)], trainer.results_folder, gif = False)
trainer.train()

# after a lot of training

sampled_seqs = diffusion.sample(batch_size = 10)
for i, seq in enumerate(sampled_seqs):
    diffusion.save_logo_plot(seq.cpu().numpy(), i, trainer.results_folder, 100)
print(sampled_seqs.shape)
seqs = one_hot_decode(sampled_seqs, characters=characters)
for seq in seqs:
    print(seq)