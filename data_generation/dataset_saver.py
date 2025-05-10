import pandas as pd
import logging


def _save_dataset_to_csv(
        timestamps,
        hour_of_day_sin,
        hour_of_day_cos,
        day_of_week_sin,
        day_of_week_cos,
        requests,
        file_name
):
    """
    Method to save the dataset as csv file.
    :param hour_of_day_sin: Sine of the hours of the day.
    :param day_of_week_cos: Cosine of the hours of the day.
    :param day_of_week_sin: Sine of the days of the week.
    :param hour_of_day_cos: Cosine of the days of the week.
    :param timestamps: List of timestamps (first column).
    :param requests: List of keys requested (second column).
    :param file_name: The name of the dataset file.
    :return:
    """
    # check the field's lengths
    if not all(len(lst) == len(timestamps) for lst in
               [hour_of_day_sin, hour_of_day_cos, day_of_week_sin, day_of_week_cos, requests]):
        raise ValueError("All input lists must have the same length.")

    # try to create and save the dataset
    try:
        # create the dataframe
        df = pd.DataFrame({
            "timestamp": timestamps,
            "hour_of_day_sin": hour_of_day_sin,
            "hour_of_day_cos": hour_of_day_cos,
            "day_of_week_sin": day_of_week_sin,
            "day_of_week_cos": day_of_week_cos,
            "key": requests
        })

        # convert the dataframe to CSV file
        df.to_csv(file_name, index=False)

        logging.info("Dataset saved correctly.")

    except Exception as e:
        raise Exception(f"An unexpected error occured while saving the dataset: {e}")