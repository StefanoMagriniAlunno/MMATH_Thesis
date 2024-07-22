import argparse
import os

from PIL import Image


def convert_png_to_pgm(input_path, output_path):
    with Image.open(input_path) as img:
        grayscale_img = img.convert("L")
        grayscale_img.save(output_path, format="PPM")


def batch_convert_png_to_pgm(input_directory, output_directory):
    for dirname, _, filenames in os.walk(input_directory):
        relative_path = os.path.relpath(dirname, input_directory)
        if not os.path.exists(os.path.join(output_directory, relative_path)):
            os.mkdir(os.path.join(output_directory, relative_path))
        for filename in filenames:
            if filename.endswith(".png"):
                filename_new = filename.replace(".png", ".pgm")
                input_path = os.path.join(input_directory, relative_path, filename)
                output_path = os.path.join(
                    output_directory, relative_path, filename_new
                )
                convert_png_to_pgm(input_path, output_path)
        print(relative_path, "DONE")


# Esempio di utilizzo
parser = argparse.ArgumentParser(
    description="This script convert a database of png images in a database with pgm images."
)
parser.add_argument(
    "input_directory", type=str, help="Directory di input con i file PNG"
)
parser.add_argument(
    "output_directory", type=str, help="Directory di output per i file PGM"
)

# Parsing degli argomenti
args = parser.parse_args()

# Accesso agli argomenti
input_directory = args.input_directory
output_directory = args.output_directory

if not os.path.exists(input_directory):
    raise FileNotFoundError(f"Input directory {input_directory} does not exist")
if not os.path.exists(output_directory):
    raise FileNotFoundError(f"Output directory {output_directory} does not exist")

batch_convert_png_to_pgm(input_directory, output_directory)
