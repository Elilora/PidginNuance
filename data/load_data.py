from huggingface_hub import hf_hub_download
import pandas as pd
from utils import logger


# Initialise data variable as None to load the data only once
data = None 
test_data = None

def load_train_data():
    """
    Loads the train dataset from Hugging Face Hub and returns it.
    If the dataset is already loaded, it returns the loaded dataset.
    """
    global data
    if data is None:
        logger.info("Loading the train dataset from Hugging Face Hub...")
        path = hf_hub_download(repo_id="WAZOBIALABS/nigerian-pidgin-voice-text",filename="wazobia_pidgin_v07_500.csv",repo_type="dataset",)
        data = pd.read_csv(path)
        logger.info("Loaded %d rows", len(data))
    return data

def load_test_data():
    """
    Loads the test dataset from Hugging Face Hub and returns it.
    If the dataset is already loaded, it returns the loaded dataset.
    """
    global test_data
    if test_data is None:
        logger.info("Loading the test dataset from Hugging Face Hub...")
        path = hf_hub_download(repo_id="WAZOBIALABS/nigerian-pidgin-eval",filename="Wazobia Labs  - Evalset.csv",repo_type="dataset",)
        test_data = pd.read_csv(path)
        logger.info("Loaded %d rows", len(test_data))
    return test_data