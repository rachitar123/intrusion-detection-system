import os
import urllib.request
import pandas as pd

def download_file(url, filename):
    print(f"Downloading {url} to {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {str(e)}")

def prepare_data(data_dir="./data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
    test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
    
    train_file = os.path.join(data_dir, "KDDTrain+.txt")
    test_file = os.path.join(data_dir, "KDDTest+.txt")
    
    if not os.path.exists(train_file):
        download_file(train_url, train_file)
    else:
        print(f"{train_file} already exists.")
        
    if not os.path.exists(test_file):
        download_file(test_url, test_file)
    else:
        print(f"{test_file} already exists.")

if __name__ == "__main__":
    prepare_data()
