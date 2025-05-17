from utils.config_utils import _get_config_value

# data configuration

# dataset
dataset_type = _get_config_value("data.distribution_type")
static_dataset_path = _get_config_value("data.static_dataset_path")
dynamic_dataset_path = _get_config_value("data.dynamic_dataset_path")

# data
num_requests = _get_config_value("data.num_requests")
num_keys = _get_config_value("data.num_keys")
first_key = _get_config_value("data.first_key")
last_key = _get_config_value("data.last_key") + 1
freq_windows = _get_config_value("data.freq_windows")
embedding_dim = _get_config_value("data.embedding_dim")

# zipf config
alpha = _get_config_value("data.alpha")
alpha_start = _get_config_value("data.alpha_start")
alpha_end = _get_config_value("data.alpha_end")
time_steps = _get_config_value("data.time_steps")

# pattern config
burst_high = _get_config_value("data.burst_high")
burst_low = _get_config_value("data.burst_low")
burst_every = _get_config_value("data.burst_every")
burst_peak = _get_config_value("data.burst_peak")
periodic_base_scale = _get_config_value("data.periodic_base_scale")
periodic_amplitude = _get_config_value("data.periodic_amplitude")

# other information
training_perc = _get_config_value("data.training_perc")
seq_len = _get_config_value("data.seq_len")

# training
learning_rate = _get_config_value("training.learning_rate")
training_batch_size = _get_config_value("training.batch_size")
training_epochs = _get_config_value("training.epochs")
optimizer_type = _get_config_value("training.optimizer")
weight_decay = _get_config_value("training.weight_decay")
momentum = _get_config_value("training.momentum")

# model
model_params = _get_config_value("model.params")
num_layers = _get_config_value("model.params.num_layers")
dropout = _get_config_value("model.params.dropout")
num_features = _get_config_value("model.num_features")
model_save_path = _get_config_value("model.model_save_path")

# validation
search_space = _get_config_value("validation.search_space")
num_folds = _get_config_value("validation.num_folds")
validation_epochs = _get_config_value("validation.epochs")
early_stopping_patience = _get_config_value("validation.early_stopping_patience")
early_stopping_delta = _get_config_value("validation.early_stopping_delta")
top_k = _get_config_value("evaluation.top_k")