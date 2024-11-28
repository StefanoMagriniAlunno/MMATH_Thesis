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
    logger.info(f"Total memory: {total_memory}")
    logger.info(f"Reserved memory: {reserved_memory}")
    logger.info(f"Allocated memory: {allocated_memory}")
    logger.info(f"Free memory: {free_memory}")
    logger.info(f"number of dimensions: {n_dimensions}")
    logger.info(f"number of centroids: {n_centroids}")
    logger.info(f"Batch size: {batch_size}")

    mu_num = torch.zeros((n_centroids,), dtype=torch.float32, device="cuda")
    mu_den = torch.zeros((n_centroids,), dtype=torch.float32, device="cuda")
    weight_synth_1_num = torch.zeros((n_centroids,), dtype=torch.float32, device="cuda")
    weight_synth_1_den = torch.zeros((n_centroids,), dtype=torch.float32, device="cuda")
    fuzzy_synth_1 = torch.zeros((n_centroids,), dtype=torch.float32, device="cuda")
    weight_synth_2_num = torch.zeros((n_centroids,), dtype=torch.float32, device="cuda")
    weight_synth_2_den = torch.zeros((n_centroids,), dtype=torch.float32, device="cuda")
    fuzzy_synth_2 = torch.zeros((n_centroids,), dtype=torch.float32, device="cuda")

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
        d_matrix = d_matrix**2

        # update mu_num
        mu_num += torch.einsum("x, xc, xc -> c", d_weights, d_matrix, d_distances)
        mu_den += torch.einsum("x, xc -> c", d_weights, d_matrix)
        weight_synth_1_num += torch.einsum("x, xc -> c", d_weights, d_matrix)
        weight_synth_1_den += torch.einsum("x -> ", d_weights)
        fuzzy_synth_1 = torch.max(fuzzy_synth_1, torch.max(d_matrix, dim=0).values)

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
        d_matrix = d_matrix**2

        # update mu_num
        mu_num += torch.einsum("x, xc, xc -> c", d_weights, d_matrix, d_distances)
        mu_den += torch.einsum("x, xc -> c", d_weights, d_matrix)
        weight_synth_2_num += torch.einsum("x, xc -> c", d_weights, d_matrix)
        weight_synth_2_den += torch.einsum("x -> ", d_weights)
        fuzzy_synth_2 = torch.max(fuzzy_synth_2, torch.max(d_matrix, dim=0).values)

    # compute final values
    mu = (mu_num / mu_den) ** (n_dimensions / 2)
    weight_synth_1 = weight_synth_1_num / weight_synth_1_den
    weight_synth_2 = weight_synth_2_num / weight_synth_2_den
    fuzzy_intersection = torch.sum(torch.minimum(fuzzy_synth_1, fuzzy_synth_2))
    fuzzy_union = torch.sum(torch.maximum(fuzzy_synth_1, fuzzy_synth_2))
    fuzzy_JaccardIndex = fuzzy_intersection / fuzzy_union

    logger.info(
        f"fuzzy parameters: {fuzzy_intersection} shared centroids, {fuzzy_union} total centroids"
    )

    # compute distance
    integrand = (
        (weight_synth_1 - weight_synth_2) / (weight_synth_1 + weight_synth_2)
    ) ** 2
    dist = (
        (1 + fuzzy_JaccardIndex) ** (-1)
        * torch.einsum("c,c->", integrand, mu)
        / torch.einsum("c->", mu)
    )
    return float(dist.item())
