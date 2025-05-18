from utils.config_utils import _get_config_value

# --------------------------------------- data config --------------------------------------- #
# distribution
seed = _get_config_value("data.distribution.seed")
distribution_type = _get_config_value("data.distribution.type")
num_requests = _get_config_value("data.distribution.num_requests")
num_keys = _get_config_value("data.distribution.num_keys")
first_key = _get_config_value("data.distribution.key_range.first_key")
last_key = _get_config_value("data.distribution.key_range.last_key") + 1
freq_windows = _get_config_value("data.distribution.freq_windows")

# access pattern
# zipf
zipf_alpha = _get_config_value("data.access_pattern.zipf.alpha")
zipf_alpha_start = _get_config_value("data.access_pattern.zipf.alpha_start")
zipf_alpha_end = _get_config_value("data.access_pattern.zipf.alpha_end")
zipf_time_steps = _get_config_value("data.access_pattern.zipf.time_steps")
# locality
locality_prob = _get_config_value("data.access_pattern.locality.prob")

# temporal pattern
# burstiness
burst_high = _get_config_value("data.temporal_pattern.burstiness.burst_high")
burst_low = _get_config_value("data.temporal_pattern.burstiness.burst_low")
burst_every = _get_config_value("data.temporal_pattern.burstiness.burst_every")
burst_peak = _get_config_value("data.temporal_pattern.burstiness.burst_peak")
# periodic
periodic_base_scale = _get_config_value("data.temporal_pattern.periodic.base_scale")
periodic_amplitude = _get_config_value("data.temporal_pattern.periodic.amplitude")

# sequence
seq_len = _get_config_value("data.sequence.len")
embedding_dim = _get_config_value("data.sequence.embedding_dim")

# dataset
training_perc = _get_config_value("data.dataset.training_perc")
static_save_path = _get_config_value("data.dataset.static_save_path")
dynamic_save_path = _get_config_value("data.dataset.dynamic_save_path")

# --------------------------------------- model config --------------------------------------- #
# general
num_features = _get_config_value("model.general.num_features")
model_save_path = _get_config_value("model.general.save_path")

# params
model_params = _get_config_value("model.params")
num_layers = _get_config_value("model.params.num_layers")
dropout = _get_config_value("model.params.dropout")

# --------------------------------------- training config --------------------------------------- #
# general
training_num_epochs = _get_config_value("training.general.num_epochs")
training_batch_size = _get_config_value("training.general.batch_size")

# optimizer
optimizer_type = _get_config_value("training.optimizer.type")
learning_rate = _get_config_value("training.optimizer.learning_rate")
weight_decay = _get_config_value("training.optimizer.weight_decay")
momentum = _get_config_value("training.optimizer.momentum")


# --------------------------------------- validation config --------------------------------------- #
# cross-validation
cv_num_folds = _get_config_value("validation.cross_validation.num_folds")
validation_num_epochs = _get_config_value("validation.cross_validation.num_epochs")

# early stopping
early_stopping_patience = _get_config_value("validation.early_stopping.patience")
early_stopping_delta = _get_config_value("validation.early_stopping.delta")

# search space
search_space = _get_config_value("validation.search_space")

# --------------------------------------- evaluation config --------------------------------------- #
top_k = _get_config_value("evaluation.top_k")