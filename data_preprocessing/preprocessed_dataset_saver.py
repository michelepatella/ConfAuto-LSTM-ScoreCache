import logging


def _save_preprocessed_dataset(df, path):
    """
    Method to save the preprocessed dataset as a csv file.
    :param df: Dataframe to save.
    :param path: Path to save the dataset.
    :return:
    """
    try:
        # save the preprocessed dataset as csv
        df.to_csv(path, index=False)
    except Exception as e:
        raise Exception(f"Error while saving preprocessed dataset: {e}")

    logging.info(f"Preprocessed dataset saved to '{path}'.")