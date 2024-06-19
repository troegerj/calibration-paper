# Standard library imports

# Third-party imports

# Local library imports


def should_model_be_validated(epoch: int, num_epochs: int, valid_interval: int) -> bool:
    is_first_epoch = epoch == 1
    is_validation_step_specified = (epoch % valid_interval) == 0
    is_last_epoch = epoch == num_epochs
    return is_first_epoch or is_validation_step_specified or is_last_epoch
