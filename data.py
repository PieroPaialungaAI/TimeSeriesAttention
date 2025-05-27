from data_utils import * 
from constants import * 
from utils import * 

class Dataset():
    def __init__(self, dataset_filepath = JSON_FILE_PATH):
        self.data_instruction = load_json(dataset_filepath)
        self.build_from_dictionary()


    def build_from_dictionary(self):
        time_steps = np.linspace(0,2*np.pi,self.data_instruction['num_points'])
        self.build_data_features()
        self.X = np.zeros((self.data_instruction['dataset_size']*2, self.data_instruction['num_points']))
        k = 0
        self.Y = np.array([0,1]*self.data_instruction['dataset_size']*2)
        #for index in range(self.data_instruction['dataset_size']):
        #    self.X[k] = 

        

    def build_data_key_features(self, key = 'loc'):
        min_value = int(self.data_instruction[f'min_{key}']*self.data_instruction['num_points'])
        max_value = int(self.data_instruction[f'max_{key}']*self.data_instruction['num_points'])
        if key in ['loc','length']:
            step_value = max(1, int(self.data_instruction[f'step_{key}']*self.data_instruction['num_points']))
        else:
            step_value = self.data_instruction[f'step_{key}']
        res = np.arange(min_value,max_value,step_value)
        res = np.random.choice(res, size = self.data_instruction['dataset_size'])
        return res
    

    def build_data_features(self):
        self.features_dict = {}
        for key in KEY_LIST:
            self.features_dict[key] = self.build_data_key_features(key)
        



