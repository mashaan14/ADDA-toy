"""Params for ADDA."""

# params for training network
num_gpu = 1
num_epochs_pre = 2 #10
log_step_pre = 20
eval_step_pre = 20
save_step_pre = 100
num_epochs = 10 #100
log_step = 100
save_step = 100
manual_seed = None


# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9
