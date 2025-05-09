import pandas as pd
import numpy as np
from utils.config_loader import load_config

def calculate_zipf_distribution_probs(keys, alpha):
    """
    Method to calculate the Zipf distribution's probabilities.
    :param keys: List of keys.
    :param alpha: Zipf distribution's parameter.
    :return: Zipf distribution's probabilities as output.
    """
    # calculate the probability of the keys according to the Zipf's distribution
    probs = 1.0 / np.power(keys, alpha)

    # normalize probabilities to make sum to 1
    probs = probs / np.sum(probs)

    return probs

def create_dataset(timestamps, requests, file_name):
    """
    Method to create the dataset (csv file).
    :param timestamps: List of timestamps (first column).
    :param requests: List of keys requested (second column).
    :param file_name: The name of the dataset file.
    :return:
    """
    # create the dataframe
    df = pd.DataFrame({'timestamp': timestamps, 'key': requests})

    # convert the dataframe to CSV file
    df.to_csv(file_name, index=False)

def generate_static_requests(n_requests, n_keys, alpha):
    """
    Method to generate static requests (with the associated timestamps).
    :param n_requests: Total number of requests.
    :param n_keys: Total number of keys.
    :param alpha: Zipf distribution's parameter.
    :return: Static requests and timestamps as output.
    """
    # calculate the probabilities
    probs = calculate_zipf_distribution_probs(np.arange(1, n_keys + 1), alpha)

    # generate requests and timestamps
    requests = np.random.choice(np.arange(1, n_keys + 1), size=n_requests, p=probs)
    timestamps = np.arange(n_requests)

    return requests, timestamps

def generate_dynamic_requests(n_requests, n_keys, alpha_values, time_steps):
    """
    Method to generate dynamic requests (with the associated timestamps).
    :param n_requests: Total number of requests.
    :param n_keys: Total number of keys.
    :param alpha_values: Zipf distribution's parameter values.
    :param time_steps: Total number of time steps.
    :return: Dynamic requests and timestamps as output.
    """
    # calculate the time step duration
    time_step_duration = n_requests // time_steps
    requests, timestamps = [], []

    # for each alpha value
    for t, alpha in enumerate(alpha_values):
        # calculate the probabilities
        probs = calculate_zipf_distribution_probs(np.arange(1, n_keys + 1), alpha)

        # generate requests and timestamps
        reqs = np.random.choice(np.arange(1, n_keys + 1), size=time_step_duration, p=probs)
        requests.extend(reqs)
        timestamps.extend(np.arange(t * time_step_duration, (t + 1) * time_step_duration))

    return requests, timestamps

def generate_static_dataset():
    """
    Method to generate static dataset following Zipf's distribution.
    :return:
    """
    # load data configuration
    config = load_config()
    data_config = config['data']
    n_requests = data_config['n_requests']
    n_keys = data_config['n_keys']

    # get the Zipf distribution's parameter
    alpha = data_config['alpha']

    # generate static requests and timestamps
    requests, timestamps = generate_static_requests(n_requests, n_keys, alpha)

    # create the static dataset
    create_dataset(timestamps, requests, data_config['static_dataset_path'])

def generate_dynamic_dataset():
    """
    Method to generate dynamic dataset following Zipf's distribution.
    :return:
    """
    # load data configuration
    config = load_config()
    data_config = config['data']
    n_requests = data_config['n_requests']
    n_keys = data_config['n_keys']

    # get the initial and final Zipf distribution's parameter
    alpha_start = data_config['alpha_start']
    alpha_end = data_config['alpha_end']

    # get the time steps
    time_steps = data_config['time_steps']

    # generate the Zipf distribution's parameter values
    alpha_values = np.linspace(alpha_start, alpha_end, time_steps)

    # generate dynamic requests and timestamps
    requests, timestamps = generate_dynamic_requests(n_requests, n_keys, alpha_values, time_steps)

    # create the dynamic dataset
    create_dataset(timestamps, requests, data_config['dynamic_dataset_path'])

def generate_zipf_dataset(distribution_type):
    """
    Method to call the (static or dynamic) zipf dataset generator.
    :param distribution_type: Zipf distribution's type (static or dynamic).
    :return:
    """
    # generate a static dataset
    if distribution_type == "static":
        generate_static_dataset()

    # generate a dynamic dataset
    elif distribution_type == "dynamic":
        generate_dynamic_dataset()

    # handle errors
    else:
        raise ValueError("Unknown distribution type.")

generate_zipf_dataset("static")
generate_zipf_dataset("dynamic")