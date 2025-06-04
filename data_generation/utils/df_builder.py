import pandas as pd
from utils.logs.log_utils import info


def create_dataframe(columns):
    """
    Method to create a dataframe.
    :param columns: The columns to create the dataframe from.
    :return: The dataframe created.
    """
    # initial message
    info("🔄 Dataset creation started...")

    try:
        # check the columns
        if any(col is None for col in columns):
            raise ValueError("❌ One or more elements in 'columns' are None.")

        # create the dataframe
        df = pd.DataFrame(columns)
    except (
            ValueError,
            TypeError
    ) as e:
        raise RuntimeError(f"❌ Error while creating the dataframe: {e}.")

    # show a successful message
    info(f"🟢 Dataframe created.")

    return df