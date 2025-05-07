import pandas as pd
import numpy as np
from utils.config_loader import load_config

def zipf_distribution(keys, alpha):
    """
    Method to generate Zipf distribution's probabilities.
    :param keys: File keys
    :param alpha: Zipf distribution's parameter
    :return: probabilities: Zipf distribution's probabilities
    """
    # calculate the probability of the keys according to Zipf's distribution
    probabilities = 1.0 / np.power(keys, alpha)

    # normalize probabilities to make sum to 1
    probabilities = probabilities / np.sum(probabilities)

    return probabilities

def create_dataset(timestamps, requests, file_name):
    """
    Method to create the dataset.
    :param timestamps: Timestamps (first column)
    :param requests: File requests (second column)
    :param file_name: The name of the dataset file
    :return:
    """
    # create the dataframe
    df = pd.DataFrame({'timestamp': timestamps, 'key': requests})

    # convert the dataframe to CSV file
    df.to_csv(file_name, index=False)

def generate_static_zipf_requests():
    """
    Method to generate static access logs according to the Zipf's distribution.
    :return:
    """
    # load data configuration
    config = load_config()['data']
    n_requests = config['n_requests']
    n_keys = config['n_keys']
    alpha = config['alpha']
    file_name = config['static_dataset_path']

    # create a list of keys
    keys = np.arange(1, n_keys + 1)

    # calculate Zipf distribution's probabilities
    probabilities = zipf_distribution(keys, alpha)

    # generate requests with timestamps
    requests = np.random.choice(keys, size=n_requests, p=probabilities)
    timestamps = np.arange(n_requests)

    # create the dataset
    create_dataset(timestamps, requests, file_name)

def generate_zipf_dynamic_requests():
    """
    Method to generate dynamic access logs according to the Zipf's distribution.
    :return:
    """
    # load data configuration
    config = load_config()['data']
    n_requests = config['n_requests']
    n_keys = config['n_keys']
    alpha_start = config['alpha_start']
    alpha_end = config['alpha_end']
    time_steps = config['time_steps']
    file_name = config['dynamic_dataset_path']

    # generate time intervals, each with a different alpha value
    alpha_values = np.linspace(alpha_start, alpha_end, time_steps)

    time_step_duration = n_requests // time_steps
    requests = []
    timestamps = []

    # generate requests for each time step
    for t in range(time_steps):

        alpha = alpha_values[t]

        # create a list of keys
        keys = np.arange(1, n_keys + 1)

        # calculate Zipf distribution's probabilities
        probabilities = zipf_distribution(keys, alpha)

        # generate random requests in this temporal interval
        start_idx = t * time_step_duration
        end_idx = (t + 1) * time_step_duration
        time_interval_requests = (
            np.random.choice(keys, size=time_step_duration, p=probabilities))

        requests.extend(time_interval_requests)
        timestamps.extend(np.arange(start_idx, end_idx))

    # create the dataset
    create_dataset(timestamps, requests, file_name)