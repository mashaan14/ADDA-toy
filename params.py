"""Params for ADDA."""

# params for training network
num_gpu = 1
num_epochs_pre = 1000 #10
save_step_pre = 100
num_epochs = 2000 #100
log_step = 100
save_step = 100
manual_seed = None


# params for optimizing models
batch_size = 32
d_lr = 2*1e-4 #1e-4
g_lr = 2*1e-4 #1e-4
