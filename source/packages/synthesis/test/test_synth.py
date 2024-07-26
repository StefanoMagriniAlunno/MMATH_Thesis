import os
import sys
from typing import List

from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_synth():
    from synthesis import synthetizer

    # cambio il formato da P2 a P5 dei file in in creando la cartella in_out/bin
    os.makedirs(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "bin"),
        exist_ok=True,
    )
    image = Image.open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.path.join("in", "file.pgm"),
        )
    ).convert("L")
    image.save(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.path.join("out", "bin", "file.pgm"),
        ),
        format="PPM",
    )

    for n_tails in range(1, 6):
        os.makedirs(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.path.join("out", "computed", str(n_tails)),
            ),
            exist_ok=True,
        )
        try:
            synthetizer(
                None,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "bin"),
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    os.path.join("out", "computed", str(n_tails)),
                ),
                "temp/list_file.txt",
                "temp/log_file.log",
                n_tails,
                8,
            )
        except SyntaxError:
            print("errore di implementazione")
            raise
        except ValueError:
            print("input non validi")
            raise
        except Exception:
            print("errore generico")
            raise

    for n_tails in range(1, 6):
        # carico l'output atteso
        v: List[int] = []
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.path.join("expected", str(n_tails)),
                "file.pgm",
            ),
            "r",
        ) as f:
            line = f.readline()
            v = [int(num) for num in line.strip().split()]
        # carico l'output prodotto
        w: List[int] = []
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.path.join("out", "computed", str(n_tails)),
                "file.pgm",
            ),
            "br",
        ) as f:
            # ogni byte è un numero uint8 (0-255) w è la lista di questi numeri
            w = list(f.read())

        assert len(v) == len(w)
        for i in range(len(v)):
            try:
                assert v[i] == w[i]
            except AssertionError:
                print(f"n_tails: {n_tails}")
                print(f"Errore in posizione {i}")
                print(f"v: {v}")
                print(f"w: {w}")
                raise
