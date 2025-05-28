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
        self.X = np.zeros((self.data_instruction['dataset_size'], self.data_instruction['num_points']))
        k = 0
        self.Y = np.array([0]*self.data_instruction['dataset_size'])
        ratio_normal_anormal = self.data_instruction['normal_anomalous_ratio']
        for idx in range(self.data_instruction['dataset_size']):
            time_steps = np.linspace(0,2*np.pi,self.data_instruction['num_points'])
            freq_idx, amp_idx = self.features_dict['freq'][idx], self.features_dict['amp'][idx]
            loc_idx, length_idx = self.features_dict['loc'][idx], self.features_dict['length'][idx]
            normal_anormal = np.random.choice([0,1], p = [ratio_normal_anormal, 1-ratio_normal_anormal])
            x_k = build_sine_wave(time_steps, frequency = freq_idx, amplitude=amp_idx)
            if normal_anormal == 1:
                x_k_modified = modify_sine(x_k, loc_idx, length_idx)
                self.X[k] = x_k_modified
                self.Y[k] = 1
            else:
                self.X[k] = x_k
            k += 1


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
        



