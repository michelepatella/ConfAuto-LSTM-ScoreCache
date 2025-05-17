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