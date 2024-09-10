'''
Helper functions used in the repository
'''
import torch 

import os
import zipfile 

from pathlib import Path

import requests

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def data_download(source : str,
                  destination: str, 
                  remove_source : bool = True) -> Path:

    '''
    Download the data (zip) from a source and unzip to desitination. 
    Can remove the source zip file. 
    '''

    data_path = Path("data/")
    image_path = data_path / destination

    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] {image_path} Creating image path..")
        image_path.mkdir(parents = True, exist_ok = True)


        target_file = Path(source).name
        with open(data_path / target_file , "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...")
            zip_ref.extractall(image_path)

        if remove_source:
            os.remove(data_path / target_file)

    return image_path




