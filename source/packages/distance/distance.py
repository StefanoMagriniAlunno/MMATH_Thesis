import logging

import numpy
import torch
import tqdm


def compute_distance(
    logger: logging.Logger,
    synth_1: numpy.ndarray,
    synth_2: numpy.ndarray,
    weights_1: numpy.ndarray,
    weights_2: numpy.ndarray,
    centroids: numpy.ndarray,
) -> float:

    # allocate the device memory
    n_dimensions = centroids.shape[1]
    n_centroids = centroids.shape[0]
    batch_size = 10000
    total_memory = torch.cuda.get_device_properties(0).total_memory
    reserved_memory = torch.cuda.memory_reserved(
        0
    )  # Memoria riservata (compresa quella già in uso)
    allocated_memory = torch.cuda.memory_allocated(0)  # Memoria effettivamente allocata
    free_memory = total_memory - reserved_memory  # Memoria libera
    logger.debug(f"Total memory: {total_memory}")
    logger.debug(f"Reserved memory: {reserved_memory}")
    logger.debug(f"Allocated memory: {allocated_memory}")
    logger.debug(f"Free memory: {free_memory}")
    logger.debug(f"number of dimensions: {n_dimensions}")
    logger.debug(f"number of centroids: {n_centroids}")
    logger.debug(f"Batch size: {batch_size}")

    # allocate cluster's values
    sigma_num = torch.zeros((n_centroids,), dtype=torch.float32, device="cuda")
    E_num = torch.zeros((n_centroids,), dtype=torch.float32, device="cuda")
    p_A = torch.zeros((n_centroids,), dtype=torch.float32, device="cuda")
    p_B = torch.zeros((n_centroids,), dtype=torch.float32, device="cuda")
    A_membership = torch.zeros((n_centroids,), dtype=torch.float32, device="cuda")
    B_membership = torch.zeros((n_centroids,), dtype=torch.float32, device="cuda")

    # copy the centroids to the device
    d_centroids = torch.tensor(centroids, dtype=torch.float32, device="cuda")

    # using data batches to compute values
    for i in tqdm.tqdm(
        range(0, synth_1.shape[0], batch_size),
        "compute values for the synthesis 1",
        leave=False,
    ):
        n_data = min(batch_size, synth_1.shape[0] - i)
        d_data = torch.tensor(
            synth_1[i : i + n_data], dtype=torch.float32, device="cuda"
        )
        d_weights = torch.tensor(
            weights_1[i : i + n_data], dtype=torch.float32, device="cuda"
        )

        # compute d_distances
        d_distances = torch.cdist(d_data, d_centroids, p=2) ** (2)

        # compute d_matrix
        d_matrix = 1 / (d_distances * torch.sum(1 / d_distances, dim=1, keepdim=True))
        d_matrix[torch.isnan(d_matrix)] = 1
        d_matrix = d_matrix / torch.sum(d_matrix, dim=1, keepdim=True)

        # update values
        sigma_num += torch.einsum("x, xc -> c", d_weights, d_matrix**2)
        E_num += torch.einsum("x, xc, xc -> c", d_weights, d_matrix**2, d_distances)
        p_A += torch.einsum("x, xc -> c", d_weights, d_matrix)
        A_membership = torch.max(A_membership, torch.max(d_matrix, dim=0).values)

    for i in tqdm.tqdm(
        range(0, synth_2.shape[0], batch_size),
        "compute values for the synthesis 2",
        leave=False,
    ):
        n_data = min(batch_size, synth_2.shape[0] - i)
        d_data = torch.tensor(
            synth_2[i : i + n_data], dtype=torch.float32, device="cuda"
        )
        d_weights = torch.tensor(
            weights_2[i : i + n_data], dtype=torch.float32, device="cuda"
        )

        # compute d_distances
        d_distances = torch.cdist(d_data, d_centroids, p=2) ** (2)

        # compute d_matrix
        d_matrix = 1 / (d_distances * torch.sum(1 / d_distances, dim=1, keepdim=True))
        d_matrix[torch.isnan(d_matrix)] = 1
        d_matrix = d_matrix / torch.sum(d_matrix, dim=1, keepdim=True)

        # update values
        sigma_num += torch.einsum("x, xc -> c", d_weights, d_matrix**2)
        E_num += torch.einsum("x, xc, xc -> c", d_weights, d_matrix**2, d_distances)
        p_B += torch.einsum("x, xc -> c", d_weights, d_matrix)
        B_membership = torch.max(B_membership, torch.max(d_matrix, dim=0).values)

    # compute final values
    E = E_num / sigma_num
    E = E / E.max()  # this value is used to avoid numerical problems
    mu = (sigma_num / (p_A + p_B)) * (E ** (n_dimensions / 2))
    p_A /= numpy.sum(weights_1)
    p_B /= numpy.sum(weights_2)

    # compute Jaccard index
    fuzzy_intersection = torch.sum(torch.minimum(A_membership, B_membership))
    fuzzy_union = torch.sum(torch.maximum(A_membership, B_membership))
    JaccardIndex = fuzzy_intersection / fuzzy_union
    logger.info(
        f"fuzzy parameters: {fuzzy_intersection} shared centroids, {fuzzy_union} total centroids"
    )

    # compute mean integral
    integrand = ((p_A - p_B) / (p_A + p_B)) ** 2
    mean_integral = torch.einsum("c,c->", integrand, mu) / torch.einsum("c->", mu)
    logger.info(f"mean integral: {mean_integral}")

    # compute distance
    dist = (1 + JaccardIndex) ** (-1) * mean_integral
    logger.info(f"distance: {dist.item()}")

    return float(dist.item())
