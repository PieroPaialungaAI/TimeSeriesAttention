from data_utils import * 
from constants import * 
from utils import * 

class Dataset():
    def __init__(self, dataset_filepath = JSON_FILE_PATH):
        self.data_instruction = load_json(dataset_filepath)
        self.build_data()


    def build_data(self):
        time_steps = np.linspace(0,2*np.pi,self.data_instruction['num_points'])
        self.build_data_features()
        self.X = np.zeros((self.data_instruction['dataset_size']*2, self.data_instruction['num_points']))
        k = 0
        self.Y = np.array([0,1]*self.data_instruction['dataset_size']*2)
        for idx in range(self.data_instruction['dataset_size']):
            time_steps = np.linspace(0,2*np.pi,self.data_instruction['num_points'])
            freq_idx, amp_idx = self.features_dict['freq'][idx], self.features_dict['amp'][idx]
            loc_idx, length_idx = self.features_dict['loc'][idx], self.features_dict['length'][idx]
            x_k = build_sine_wave(time_steps, frequency = freq_idx, amplitude=amp_idx) 
            self.X[k] = x_k
            self.X[k+1] = x_k
            k += 2


        

    def build_data_key_features(self, key = 'loc'):

        if key in ['loc','length']: #Filtering the integer cases
            min_value = int(self.data_instruction[f'min_{key}']*self.data_instruction['num_points'])
            max_value = int(self.data_instruction[f'max_{key}']*self.data_instruction['num_points'])
            step_value = max(1, int(self.data_instruction[f'step_{key}']*self.data_instruction['num_points']))
        else:
            min_value = self.data_instruction[f'min_{key}']
            max_value = self.data_instruction[f'max_{key}']
            step_value = self.data_instruction[f'step_{key}']
        res = np.arange(min_value,max_value,step_value)
        res = np.random.choice(res, size = self.data_instruction['dataset_size'])
        return res
    

    def build_data_features(self):
        self.features_dict = {}
        for key in KEY_LIST:
            self.features_dict[key] = self.build_data_key_features(key)
        



