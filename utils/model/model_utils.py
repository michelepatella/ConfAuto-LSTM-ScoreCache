import numpy as np
import torch
from sklearn.utils import compute_class_weight
from utils.logs.log_utils import info, debug


def calculate_class_weights(
        targets,
        config_settings
):
    """
    Method to calculate the class weights.
    :param targets: The targets for which to calculate the class weights.
    :param config_settings: The configuration settings.
    :return: The class weights calculated.
    """
    # initial message
    info("🔄 Class weights calculation started...")

    try:
        # debugging
        debug(f"⚙️ Number of classes: {config_settings.num_keys}.")

        # be sure targets is a numpy array and shift them
        targets = targets.cpu().numpy() \
            if isinstance(targets, torch.Tensor)\
            else targets

        # get the classes appearing in target list
        present_classes = np.unique(targets)

        # debugging
        debug(f"⚙️ Present classes: {present_classes}.")

        # compute the class weights
        computed_weights = compute_class_weight(
            class_weight="balanced",
            classes=present_classes,
            y=targets
        )

        # initialize weights to 1.0
        class_weights = np.ones(
            config_settings.num_keys,
            dtype=np.float32
        )

        # update weights for appearing classes
        for cls, weight in zip(
                present_classes,
                computed_weights
        ):
            class_weights[cls] = weight

    except (
            ValueError,
            TypeError,
            IndexError
    ) as e:
        raise RuntimeError(f"❌ Error while calculating the class weights: {e}.")

    # show a successful message
    info("🟢 Class weights calculated.")

    return class_weights


def build_optimizer(
        model,
        learning_rate,
        config_settings
):
    """
    Method to build the optimizer.
    :param model: Model for which the optimizer will be built.
    :param learning_rate: Learning rate.
    :param config_settings: The configuration settings.
    :return: The created optimizer.
    """
    # initial message
    info("🔄 Optimizer building started...")

    # debugging
    debug(f"⚙️ Learning rate: {learning_rate}.")
    debug(f"⚙️ Optimizer type: {config_settings.optimizer_type}.")

    try:
        # define the optimizer
        if config_settings.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate
            )
        elif config_settings.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=config_settings.weight_decay
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=config_settings.momentum
            )
    except (
            ValueError,
            TypeError,
            UnboundLocalError,
            KeyError,
            AssertionError
    ) as e:
        raise RuntimeError(f"❌ Error while building optimizer: {e}.")

    # show a successful message
    info("🟢 Optimizer building completed.")

    return optimizer