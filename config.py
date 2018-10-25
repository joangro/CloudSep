# DATABASE
db_path='db/zenodo.org/record/1117372/files/'
log_dir = 'stft/log/'

# Algorithm regulariation values
alpha = 0.001
beta  = 0.01
beta_voc = 0.03

# Hyperparameters
input_features = 513
output_features = 66
highway_layers = 4
highway_units = 128
num_conv_layers = 8
conv_filters = 128

# conv_activation = tf.nn.relu
fs = 44100
comp_mode = 'mfsc'
