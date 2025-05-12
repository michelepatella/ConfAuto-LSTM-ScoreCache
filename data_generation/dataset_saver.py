import pandas as pd
import logging


def _save_dataset_to_csv(columns, file_name):
    """
    Method to save the dataset as csv file.
    :param columns: The columns of the dataset to be saved.
    :param file_name: The name of the dataset file.
    :return:
    """
    # check the column's lengths
    lengths = [len(v) for v in columns.values()]
    if len(set(lengths)) != 1:
        raise ValueError("All columns must have the same length.")

    try:
        # create the dataframe
        df = pd.DataFrame(columns)

        # convert the dataframe to CSV file
        df.to_csv(file_name, index=False)

        # show a successful message
        logging.info(f"🟢Dataset saved to '{file_name}'.")

    except Exception as e:
        raise Exception(f"Error while saving the dataset: {e}")