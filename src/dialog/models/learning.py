"""TODO: Add docstring."""

import numpy as np


def update_theta_and_beta_parameters(
    latent_driver_map: np.ndarray,
    latent_driver_samples: list,
    thetas: np.ndarray,
    betas: np.ndarray,
    persistent_chain: np.ndarray,
    learning_rate: float,
    lambda_theta: float,
    lambda_beta: float,
    momentum: float,
) -> tuple:
    """TODO: Add docstring."""
    num_samples = latent_driver_map.shape[0]
    curr_latent_drivers = np.mean(np.stack(latent_driver_samples, axis=0), axis=0)
    new_persistent_chain = (
        momentum * persistent_chain + (1 - momentum) * curr_latent_drivers
    )
    empirical_mean = latent_driver_map.mean(axis=0)
    model_mean = new_persistent_chain.mean(axis=0)

    theta_grad = empirical_mean - model_mean

    empirical_corr = latent_driver_map.T @ latent_driver_map / num_samples
    model_corr = new_persistent_chain.T @ new_persistent_chain / num_samples
    beta_grad = empirical_corr - model_corr

    new_thetas = thetas + learning_rate * theta_grad
    new_betas = betas + learning_rate * beta_grad

    def _soft_threshold(x: np.array, thresh: float) -> np.array:
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)

    new_thetas = _soft_threshold(new_thetas, learning_rate * lambda_theta)
    new_betas = _soft_threshold(new_betas, learning_rate * lambda_beta)

    return new_thetas, new_betas, new_persistent_chain
