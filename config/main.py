from utils.config_utils import _get_config_value

# data configuration

# general data config
num_requests = _get_config_value("data.num_requests")
num_keys = _get_config_value("data.num_keys")
first_key = _get_config_value("data.first_key")
last_key = _get_config_value("data.last_key") + 1
freq_windows = _get_config_value("data.freq_windows")

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

# model
model_params = _get_config_value("model.params")
model_save_path = _get_config_value("model.model_save_path")

# validation
search_space = _get_config_value("validation.search_space")
num_folds = _get_config_value("validation.num_folds")
validation_epochs = _get_config_value("validation.epochs")