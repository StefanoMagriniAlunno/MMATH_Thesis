import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_clust():
    from clustering import fcm

    n_data = 100000
    n_dimensions = 10
    n_centroids = 20
    tollerance = 0.001

    # genero un set di dati sintetici
    synth = np.random.rand(n_data, n_dimensions).astype(np.float32)

    # salvo i dati in un file temporaneo come float32 binario (i dati da clusterizzare)
    with open("temp/synth", "bw") as f:
        f.write(synth.tobytes())

    # estraggo un campione di righe da synth
    synth_sample = np.random.choice(synth.shape[0], n_centroids, replace=False)
    synth_sample = synth[synth_sample]

    # salvo il campione in un file temporaneo come float32 binario (i centroidi iniziali)
    with open("temp/synth_sample", "bw") as f:
        f.write(synth_sample.tobytes())

    # genero dei pesi per ogni dato
    synth_weights = np.random.rand(synth.shape[0]).astype(np.float32)

    # salvo i pesi in un file temporaneo come float32 binario
    with open("temp/synth_weights", "bw") as f:
        f.write(synth_weights.tobytes())

    # libero la ram (fondamentale per evitare memory error)
    del synth
    del synth_sample
    del synth_weights

    # genero il logger
    logger = logging.getLogger("clustering")
    logger.setLevel(logging.DEBUG)

    # eseguo il clustering fcm
    fcm(
        logger,
        "temp/synth",
        "temp/synth_weights",
        "temp/synth_sample",
        "temp/centroids",
        n_dimensions,
        tollerance,
        "logs/fcm.log",
    )

    # eseguo il clustering usando torch
    import torch

    # setto il device
    device = torch.device("cuda")
    # carico i dati
    tensor_data = torch.tensor(
        np.fromfile("temp/synth", dtype=np.float32).reshape(-1, n_dimensions),
        device=device,
    )
    # carico i centroidi iniziali
    tensor_centroids = torch.tensor(
        np.fromfile("temp/synth_sample", dtype=np.float32).reshape(-1, n_dimensions),
        device=device,
    )
    # carico i pesi
    tensor_weights = torch.tensor(
        np.fromfile("temp/synth_weights", dtype=np.float32), device=device
    )

    delta_update = 1
    while delta_update > 0.1:
        # calcolo la matrice delle distanze a 2 a 2 tra dati e centroidi
        distances = torch.cdist(tensor_data, tensor_centroids) ** (2)
        # calcolo la matrice di membership
        U2 = 1 / (distances * torch.sum(1 / distances, dim=1, keepdim=True))

        # nelle righe dove ci sono nan metto 1
        U2[torch.isnan(U2)] = 1
        # normalizzo ogni riga nuovamente
        U2 = U2 / torch.sum(U2, dim=1, keepdim=True)

        U2 = U2**2
        U2 = U2 * tensor_weights[:, None]

        # calcolo i nuovi centroidi
        tensor_new_centroids = (
            torch.matmul(U2.T, tensor_data) / torch.sum(U2, dim=0, keepdim=True).T
        )

        # calcolo l'update
        delta_update = torch.norm(tensor_new_centroids - tensor_centroids)

        # aggiorno i centroidi
        tensor_centroids = tensor_new_centroids

    # confronto tensor_centroids con i centroidi calcolati da fcm
    centroids = np.fromfile("temp/centroids", dtype=np.float32).reshape(
        -1, n_dimensions
    )

    err = np.linalg.norm(tensor_centroids.cpu().numpy() - centroids)

    assert err / n_dimensions < 0.01, f"Errore: {err/n_dimensions}"


if __name__ == "__main__":
    test_clust()
