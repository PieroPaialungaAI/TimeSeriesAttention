from data_utils import * 
from constants import * 
from utils import * 

class Dataset():
    def __init__(self, dataset_filepath = JSON_FILE_PATH):
        self.data_instruction = load_json(dataset_filepath)
        self.build_from_dictionary()


    def build_from_dictionary(self):
        X = np.linspace(0,2*np.pi,self.data_instruction['num_points'])
        min_loc = int(self.data_instruction['min_loc']*self.data_instruction['num_points'])
        max_loc = int(self.data_instruction['max_loc']*self.data_instruction['num_points'])
        step_loc = max(1, int(self.data_instruction['step_loc']*self.data_instruction['num_points']))
        self.possible_locs = np.arange(min_loc,max_loc,step_loc)

        



