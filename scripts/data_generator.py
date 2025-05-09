import pandas as pd
import numpy as np
from utils.config_loader import load_config

def zipf_distribution(keys, alpha):
    """
    Method to generate Zipf distribution's probabilities.
    :param keys: List of keys
    :param alpha: Zipf distribution's parameter
    :return: Zipf distribution's probabilities
    """
    # calculate the probability of the keys according to Zipf's distribution
    probabilities = 1.0 / np.power(keys, alpha)

    # normalize probabilities to make sum to 1
    probabilities = probabilities / np.sum(probabilities)

    return probabilities

def create_dataset(timestamps, requests, file_name):
    """
    Method to create the dataset csv file.
    :param timestamps: List of timestamps (first column)
    :param requests: List of keys requested (second column)
    :param file_name: The name of the dataset file
    :return:
    """
    # create the dataframe
    df = pd.DataFrame({'timestamp': timestamps, 'key': requests})

    # convert the dataframe to CSV file
    df.to_csv(file_name, index=False)

def generate_zipf_dataset(distribution_type):
    """
    Method to generate the (static or dynamic) data for the dataset.
    :param distribution_type: Zipf distribution's type (static or dynamic)
    :return:
    """
    # load data configuration
    config = load_config()['data']
    n_requests = config['n_requests']
    n_keys = config['n_keys']

    # generate a static distribution
    if distribution_type == "static":

        # get the Zipf distribution's parameter
        alpha = config['alpha']

        # generate keys
        keys = np.arange(1, n_keys + 1)

        # get the probabilities
        probabilities = zipf_distribution(keys, alpha)

        # generate requests and timestamps
        requests = np.random.choice(keys, size=n_requests, p=probabilities)
        timestamps = np.arange(n_requests)

        # create the static dataset
        create_dataset(timestamps, requests, config['static_dataset_path'])

    # generate a dynamic distribution
    elif distribution_type == "dynamic":

        # get the start and final Zipf distribution's parameter
        alpha_start = config['alpha_start']
        alpha_end = config['alpha_end']

        # get the time steps
        time_steps = config['time_steps']

        # generate the Zipf distribution's parameter values within the predefined range
        alpha_values = np.linspace(alpha_start, alpha_end, time_steps)

        # define the step duration
        time_step_duration = n_requests // time_steps

        # generate dynamic requests
        requests, timestamps = [], []
        for t, alpha in enumerate(alpha_values):
            # generate keys
            keys = np.arange(1, n_keys + 1)

            # get probabilities
            probabilities = zipf_distribution(keys, alpha)

            # generate requests and timestamps
            reqs = np.random.choice(keys, size=time_step_duration, p=probabilities)
            requests.extend(reqs)
            timestamps.extend(np.arange(t * time_step_duration, (t + 1) * time_step_duration))

        # create the dynamic dataset
        create_dataset(timestamps, requests, config['dynamic_dataset_path'])

generate_zipf_dataset("static")
generate_zipf_dataset("dynamic")