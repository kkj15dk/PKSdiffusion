from denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D, Trainer1D, save_logo_plot
from utils import *

seed = 42
set_seed(seed) # set the random seed
print("seed set as " + str(seed))

model = Unet1D( # This UNET model cannot take in odd length inputs...
    dim = 128, # 64
    dim_mults = (1, 2, 4, 8),
    channels = 21,
    learned_sinusoidal_cond=True,
    random_fourier_features=True,
)

print("Model parameters: ", count_parameters(model))

test = False
varying_length = True
varying_length_resolution = 8

# aa_file = 'test.fa'
# labels_file = 'labels_test.json'

aa_file = "NRPSs_mid-1800.fa"
labels_file = 'labels.json'
characters="ACDEFGHIKLMNPQRSTVWY-"

if not test:
    seqs, max_len = load_fasta(aa_file, labels_file, characters, varying_length=varying_length, varying_length_resolution=varying_length_resolution)

if test:
    seqs = random_aa_seq(1000)
    write_fasta(seqs, 'test.fa')
    aa_file = 'test.fa'
    labels_file = 'labels_test.json'
    seqs, max_len = load_fasta(aa_file, labels_file, characters, varying_length=varying_length, varying_length_resolution = varying_length_resolution)

    print("Test of one_hot_encode and one_hot_decode:")
    for i in range(10):
        print(seqs[i])
    aa_OHE = one_hot_encode(seqs[0][0], characters=characters, max_len=max_len)
    print(one_hot_decode(aa_OHE, characters=characters))

diffusion = GaussianDiffusion1D(
    model,
    max_seq_length = max_len,
    timesteps = 1000,
    # objective = 'pred_noise',
    # objective = 'pred_x0', 
    objective = 'pred_v',
    beta_schedule = 'cosine',
    # beta_schedule = 'linear',
    auto_normalize=True,
)

# Create a Dataset
dataset = MyIterDataset(OHEAAgen, seqs, len(seqs), characters, max_len, varying_length=varying_length, varying_length_resolution = varying_length_resolution)
# plot_dataset(dataset)

# Or using trainer
if test:
    num_classes = 2
    samples = [(cl,g) for cl in range(num_classes + 1) for g in [0, 0.5, 2]]
else:
    samples = [(cl,g) for cl in [1, 2] for g in [0, 2]]
    # num_classes = 20
    # samples = [(cl,g) for cl in range(num_classes) for g in [0, 0.1, 1, 4, 10]]

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 64,
    train_lr = 2e-5, # 8e-5,
    train_num_steps = 1000000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    save_and_sample_every = 100000,
    results_folder="./resultsNRPS_masked",
    samples=samples,
    sample_len=1600,
    labels_file=labels_file,
    characters=characters,
)
# trainer.load("10")
diffusion.visualize_diffusion(next(iter(dataset)), [10*i for i in range(100)], trainer.results_folder, gif = False)
trainer.train()

# after a lot of training

# diffusion.sample_gif([(0,10)], seq_len = 1800, folder = trainer.results_folder, num_processes = 4, time_resolution = 50, ylim = (-3, 7), dpi = 20)
# diffusion.sample_gif([(1,10)], seq_len = 800, folder = trainer.results_folder, num_processes = 4, time_resolution = 50, ylim = (-3, 7), dpi = 20)
# diffusion.sample_gif([(2,10)], seq_len = 160, folder = trainer.results_folder, num_processes = 4, time_resolution = 50, ylim = (-3, 7), dpi = 20)

# sampled_seqs = diffusion.sample(samples = samples, seq_length=40)
# for i, sample in enumerate(samples):
#     # if sample[1] == 10: # making logo of all with guide_w = 10
#     save_logo_plot(sampled_seqs[i].cpu().numpy(), f'cl_{sample[0]}_w_{sample[1]}', f'{trainer.results_folder}/samples40', 100)

# sampled_seqs = diffusion.sample(samples = samples, seq_length=32)
# for i, sample in enumerate(samples):
#     # if sample[1] == 10: # making logo of all with guide_w = 10
#     save_logo_plot(sampled_seqs[i].cpu().numpy(), f'cl_{sample[0]}_w_{sample[1]}', f'{trainer.results_folder}/samples32', 100)

# sampled_seqs = diffusion.sample(samples = samples, seq_length=24)
# for i, sample in enumerate(samples):
#     # if sample[1] == 10: # making logo of all with guide_w = 10
#     save_logo_plot(sampled_seqs[i].cpu().numpy(), f'cl_{sample[0]}_w_{sample[1]}', f'{trainer.results_folder}/samples24', 100)
