import torch
from utils.logs.log_utils import info, debug


def extract_targets_from_dataloader(data_loader):
    """
    Method to extract the targets from the data loader.
    :param data_loader: The data loader from which to extract the targets.
    :return: All the extracted targets.
    """
    # initial message
    info("🔄 Target extraction from loader started...")

    try:
        all_targets = []
        # extract targets from data loader
        for _, _, targets in data_loader:
            all_targets.append(targets)
    except (
            TypeError,
            ValueError,
            IndexError
    ) as e:
        raise RuntimeError(f"❌ Error while extracting targets from loader: {e}.")

    # debugging
    debug(f"⚙️ Target extracted: {all_targets}.")

    # show a successful message
    info("🟢 Target extracted from loader.")

    return torch.cat(all_targets)