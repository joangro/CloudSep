# DATABASE
db_path='db/zenodo.org/record/1117372/files/'

alpha = 0.001
beta  = 0.01
beta_voc = 0.03

channels = 2
features = 513
num_epochs = 50
batches_per_epoch_train = 2

max_len = 3939892
channels = 2
features = 513

split = 0.9

# Hyperparameters
num_epochs = 8000
batches_per_epoch_train = 50
batches_per_epoch_val = 50
batch_size = 5
samples_per_file = 1
max_phr_len = 30
input_features = 513
lstm_size = 128
output_features = 66
highway_layers = 4
highway_units = 128
init_lr = 1
num_conv_layers = 8
conv_filters = 128
num_ch_out_hor = 50
num_ch_out_ver = 30
# conv_activation = tf.nn.relu
dropout_rate = 0.0
projection_size = 3
fs = 44100
comp_mode = 'mfsc'
dn_num_epochs = 4000
print_every = 1
save_every = 100
