from config.data_params_checker import _check_distribution_params, _check_access_pattern_params, _check_sequence_params, _check_dataset_params
from config.model_params_checker import _check_general_model_params, _check_model_params
from config.training_params_checker import _check_general_training_params, _check_optimizer_params
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

# check distribution params
_check_distribution_params(
    seed,
    distribution_type,
    num_requests,
    num_keys,
    first_key,
    last_key,
    freq_windows
)

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

# check access pattern params
_check_access_pattern_params(
    zipf_alpha,
    zipf_alpha_start,
    zipf_alpha_end,
    zipf_time_steps,
    locality_prob,
    burst_high,
    burst_low,
    burst_every,
    burst_peak,
    periodic_base_scale,
    periodic_amplitude
)

# sequence
seq_len = _get_config_value("data.sequence.len")
embedding_dim = _get_config_value("data.sequence.embedding_dim")

# check sequence params
_check_sequence_params(
    seq_len,
    embedding_dim,
    num_requests
)

# dataset
training_perc = _get_config_value("data.dataset.training_perc")
static_save_path = _get_config_value("data.dataset.static_save_path")
dynamic_save_path = _get_config_value("data.dataset.dynamic_save_path")

# check dataset params
_check_dataset_params(
    training_perc,
    static_save_path,
    dynamic_save_path
)

# --------------------------------------- model config --------------------------------------- #
# general
num_features = _get_config_value("model.general.num_features")
model_save_path = _get_config_value("model.general.save_path")

# check general params
_check_general_model_params(
        num_features,
        model_save_path
)

# params
model_params = _get_config_value("model.params")
hidden_size = _get_config_value("model.params.hidden_size")
num_layers = _get_config_value("model.params.num_layers")
bias = _get_config_value("model.params.bias")
batch_first = _get_config_value("model.params.batch_first")
dropout = _get_config_value("model.params.dropout")
bidirectional = _get_config_value("model.params.bidirectional")
proj_size = _get_config_value("model.params.proj_size")

# check model params
_check_model_params(
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        bidirectional,
        proj_size
)

# --------------------------------------- training config --------------------------------------- #
# general
training_num_epochs = _get_config_value("training.general.num_epochs")
training_batch_size = _get_config_value("training.general.batch_size")

# check general training params
_check_general_training_params(
        training_num_epochs,
        training_batch_size
)

# optimizer
optimizer_type = _get_config_value("training.optimizer.type")
learning_rate = _get_config_value("training.optimizer.learning_rate")
weight_decay = _get_config_value("training.optimizer.weight_decay")
momentum = _get_config_value("training.optimizer.momentum")

# check optimizer params
_check_optimizer_params(
        optimizer_type,
        learning_rate,
        weight_decay,
        momentum
)

# --------------------------------------- validation config --------------------------------------- #
# cross-validation
cv_num_folds = _get_config_value("validation.cross_validation.num_folds")
validation_num_epochs = _get_config_value("validation.cross_validation.num_epochs")

# early stopping
early_stopping_patience = _get_config_value("validation.early_stopping.patience")
early_stopping_delta = _get_config_value("validation.early_stopping.delta")

# search space
search_space = _get_config_value("validation.search_space")
hidden_size_range = _get_config_value("validation.search_space.model.params.hidden_size_range")
num_layers_range = _get_config_value("validation.search_space.model.params.num_layers_range")
dropout_range = _get_config_value("validation.search_space.model.params.dropout_range")
learning_rate_range = _get_config_value("validation.search_space.training.learning_rate_range")

# --------------------------------------- evaluation config --------------------------------------- #
top_k = _get_config_value("evaluation.top_k")

# --------------------------------------- testing config --------------------------------------- #
testing_batch_size = _get_config_value("testing.general.batch_size")