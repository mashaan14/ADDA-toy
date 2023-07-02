"""Params for ADDA."""

# params for training network
num_gpu = 1
num_epochs_pre = 1000
save_step_pre = 100
num_epochs = 1000
log_step = 100
save_step = 100
manual_seed = None


# params for optimizing models
batch_size = 30
d_lr = 1e-4 #1e-4
g_lr = 1e-4 #1e-4
beta1 = 0.5
beta2 = 0.9
