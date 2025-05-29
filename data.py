from data_utils import * 
from constants import * 
from utils import * 
from torch_data import SineWaveTorchDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader



class Dataset():
    def __init__(self, dataset_filepath = JSON_FILE_PATH):
        self.data_instruction = load_json(dataset_filepath)
        self.build_data()
        self.batch_size = None


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


    def to_torch(self, batch_size = 32):
        self.torch_data = SineWaveTorchDataset(self.X, self.Y)
        self.batch_size = batch_size
        return self.torch_data
    

    def train_test_split(self, test_size = 0.2, torch_data = True):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=test_size, random_state=42)
        if torch_data:
            self.train_torch_data = SineWaveTorchDataset(self.X_train, self.Y_train)
            self.test_torch_data =  SineWaveTorchDataset(self.X_test, self.Y_test)
            if self.batch_size is None:
                self.batch_size = DEFAULT_BATCH_SIZE
            self.train_loader = DataLoader(self.train_torch_data, batch_size = DEFAULT_BATCH_SIZE)
            self.test_loader = DataLoader(self.test_torch_data, batch_size = DEFAULT_BATCH_SIZE)
        

    def train_val_test_split(self, val_size=0.1, test_size=0.2, torch_data=True, batch_size=None):
        """
        Splits the dataset into train, val, and test sets.
        
        Parameters:
            val_size (float): Fraction of data to use for validation.
            test_size (float): Fraction of data to use for testing.
            torch_data (bool): Whether to return torch Dataset and DataLoader.
            batch_size (int): Batch size for DataLoaders. Defaults to self.batch_size or DEFAULT_BATCH_SIZE.
        """
        if batch_size is not None:
            self.batch_size = batch_size
        if self.batch_size is None:
            self.batch_size = DEFAULT_BATCH_SIZE

        # Step 1: train + val vs test
        X_trainval, X_test, Y_trainval, Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=42
        )

        # Step 2: train vs val
        val_ratio_adjusted = val_size / (1 - test_size)  # rescale for remaining data
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_trainval, Y_trainval, test_size=val_ratio_adjusted, random_state=42
        )

        # Save raw splits
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.Y_train, self.Y_val, self.Y_test = Y_train, Y_val, Y_test

        if torch_data:
            self.train_torch_data = SineWaveTorchDataset(X_train, Y_train)
            self.val_torch_data   = SineWaveTorchDataset(X_val, Y_val)
            self.test_torch_data  = SineWaveTorchDataset(X_test, Y_test)

            self.train_loader = DataLoader(self.train_torch_data, batch_size=self.batch_size, shuffle=True)
            self.val_loader   = DataLoader(self.val_torch_data, batch_size=self.batch_size, shuffle=False)
            self.test_loader  = DataLoader(self.test_torch_data, batch_size=self.batch_size, shuffle=False)




        



