from data_utils import * 
from constants import * 
from utils import * 

class Dataset():
    def __init__(self, dataset_filepath = JSON_FILE_PATH):
        self.data_json = load_json(dataset_filepath)


