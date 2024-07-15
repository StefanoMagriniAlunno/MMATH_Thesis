import os
import sys

from invoke import task  # type: ignore


@task
def directories(c):
    """Create directories"""
    data_dir = os.path.join(os.getcwd(), "data")
    # os.makedirs(os.path.join(data_dir, 'images'), exist_ok=True)
    # os.makedirs(os.path.join(data_dir, 'audios'), exist_ok=True)
    # os.makedirs(os.path.join(data_dir, 'videos'), exist_ok=True)
    # os.makedirs(os.path.join(data_dir, 'models'), exist_ok=True)
    # os.makedirs(os.path.join(data_dir, 'repo'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "db"), exist_ok=True)


@task
def build(c):
    """Build packages"""
    c.run("cd source && make")
    # sposto il contenuto di source/lib in lib/python3.10/site-packages
    c.run(
        "cp "
        + r"source/lib/"
        + "* "
        + os.path.join(os.getenv("VIRTUAL_ENV"), r"lib/python3.10/site-packages")
    )


@task
def install(c):
    """Install packages for development"""
    list_packages = [
        "numpy scipy pandas scikit-learn",  # scientific computing
        "matplotlib seaborn plotly",  # plotting
        "tqdm colorama",  # utilities
        "torch torchvision torchaudio",  # gpu computing
    ]

    for package in list_packages:
        c.run(f"{sys.executable} -m pip install {package}")


@task
def download(c):
    """Download datasets in data/"""
