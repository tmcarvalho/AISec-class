import os
import sys
import re
import subprocess
import tarfile
import zipfile
import urllib.request
import yaml
import pandas as pd
import numpy as np


def run_uv_install():
    """
    Installs required packages using `uv pip install`.
    """
    pkgs = [
        "datasets==2.21.0",
        "transformers==4.44.2",
        "torch==2.4.1",
        "torchvision==0.19.1",
        "torchaudio"
    ]

    try:
        subprocess.run(["uv", "pip", "install"] + pkgs, check=True)
    except subprocess.CalledProcessError as e:
        print("uv pip install failed.")
        print(e)



def remove_cuda_packages(requirements_path: str):
    """
    Removes CUDA-related packages from requirements.txt.
    Deletes entries containing:
        nvidia-, cudnn, cublas, cuda, cu11, triton
    """
    if not os.path.exists(requirements_path):
        print(f"requirements.txt not found: {requirements_path}")
        return

    CUDA_PATTERNS = [
        r"nvidia-.*",
        r".*triton.*",
    ]

    with open(requirements_path, "r") as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if any(re.match(pattern, stripped) for pattern in CUDA_PATTERNS):
            continue
        cleaned_lines.append(line)

    # Write back cleaned file
    with open(requirements_path, "w") as f:
        f.writelines(cleaned_lines)



def clone_and_setup_repo(user_name: str, repo_name: str = "ml_privacy_meter"):
    """
    Clones a forked GitHub repo and sets up the environment path.

    Args:
        user_name (str): Your GitHub username containing the fork.
        repo_name (str): The repository name (default: ml_privacy_meter)
    """

    repo_url = f"https://github.com/{user_name}/{repo_name}.git"
    clone_dir = os.path.join(os.getcwd(), repo_name)

    if not os.path.exists(clone_dir):
        subprocess.run(["git", "clone", repo_url], check=True)
    else:
        print("Repo already exists. Skipping clone.")

    # Add repo to PYTHONPATH
    if clone_dir not in sys.path:
        sys.path.append(clone_dir)

    # Change working directory
    os.chdir(clone_dir)
    print(f"Current working directory: {os.getcwd()}")

    # Install packages
    run_uv_install()

    # Clean CUDA packages from requirements.txt
    remove_cuda_packages("requirements.txt")

    # Install using uv
    try:
        subprocess.run(["uv", "pip", "install", "-r", "requirements.txt", "--index-strategy", "unsafe-best-match"], check=True)
    except subprocess.CalledProcessError as e:
        print("uv pip install failed.")
        print(e)


def write_config_yaml(config_dir: str, filename: str = "config.yaml"):
    """
    Automatically writes a YAML file into a directory.
    Creates the directory if it does not exist.
    """
    os.makedirs(config_dir, exist_ok=True)
    filepath = os.path.join(config_dir, filename)

    config = {
        "run": {
            "random_seed": 12345,
            "log_dir": "demo_locations",
            "time_log": True,
            "num_experiments": 1,
        },

        "audit": {
            "privacy_game": "privacy_loss_model",
            "algorithm": "RMIA",
            "num_ref_models": 1,
            "device": "cpu",
            "report_log": "report_rmia",
            "batch_size": 5000,
            # "data_size": 10000 
        },

        "train": {
            "model_name": "mlp",
            "device": "cpu",
            "batch_size": 256,
            "optimizer": "SGD",
            "learning_rate": 0.1,
            "weight_decay": 0,
            "epochs": 100,
        },
        
        "data": {
            "dataset": "locations",
            "data_dir": "data_locations",
        }
    }

    with open(filepath, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    return filepath


def download_file(url, compression, compressed_file, temp_dir_name):
    urllib.request.urlretrieve(url, compressed_file)

    if compression == 'tar':
        with tarfile.open(compressed_file, 'r:gz') as tar:
            tar.extractall(temp_dir_name, filter='fully_trusted')
        os.remove(compressed_file)

    elif compression == 'zip':
        with zipfile.ZipFile(compressed_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir_name)
        os.remove(compressed_file)


def add_shape(num_features, num_classes, dataset_name):
    """
    Adds new shape to INPUT_OUTPUT_SHAPE in utils.py
    if it is not already present.
    """
    utils_path = "models/utils.py"
    if not os.path.exists(utils_path):
        print(f"utils.py not found: {utils_path}")
        return

    with open(utils_path, "r") as f:
        content = f.read()

    if dataset_name == 'locations':
        num_classes = num_classes+1

    # If already exists, skip
    if f'"{dataset_name}"' in content:
        print(f"'{dataset_name}' entry already exists. No changes made.")
        return
    
    # The line to insert
    insert_line = f'    "{dataset_name}": [{num_features}, {num_classes}],'

    # Regex: insert after the first line of the dictionary
    pattern = r"(INPUT_OUTPUT_SHAPE\s*=\s*\{\s*)"
    new_content = re.sub(pattern, r"\1\n" + insert_line, content, count=1)

    with open(utils_path, "w") as f:
        f.write(new_content)

    print("utils.py updated successfully.\n")



if __name__ == "__main__":
    # TODO: CHANGE THIS to your GitHub username
    github_user = "YOUR_USERNAME"

    clone_and_setup_repo(github_user)
    write_config_yaml("configs", "locations.yaml")

    # Download locations dataset
    url = 'https://github.com/privacytrustlab/datasets/raw/refs/heads/master/dataset_location.tgz'
    temp_dir = 'locations_dir'
    download_file(url, 'tar', 'dataset_location.tgz', temp_dir)
    df = pd.read_csv(os.path.join(temp_dir, 'bangkok'), header=None)
    df['Label'] = df.pop(0) # move label to last column

    # Load configs
    configs = "configs/locations.yaml"
    with open(configs, "rb") as f:
            configs = yaml.load(f, Loader=yaml.Loader)

    df = df.to_numpy()
    y = df[:, -1]
    X = df[:, :-1].astype(np.float32)

    # Number of features
    num_features = X.shape[1]

    # Number of classes (unique targets)
    num_classes = len(np.unique(y))

    # Dataset name
    dataset_name = configs["data"]["dataset"]

    add_shape(num_features, num_classes, dataset_name)
