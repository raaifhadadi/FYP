# China Brugada ECGs DataSet class
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class ChinaDataset(Dataset):
    def __init__(self, root_dir, split='train', test_size=0.2, random_state=None, transform=None):
        
        self.root_dir = root_dir
        self.csv_file = os.path.join(root_dir, 'DAT_China_ECGs.csv')
        
        # Load annotations and scp_statements.csv for diagnostic aggregation
        self.annotations = pd.read_csv(self.csv_file, index_col="id")
        
        # Split the annotations DataFrame into train and test sets
        self.annotations_train, self.annotations_test = train_test_split(self.annotations, test_size=test_size, random_state=random_state)
        
        self.transform = transform
        
        if split == 'train':
            self.annotations = self.annotations_train
        elif split == 'test':
            self.annotations = self.annotations_test
        else:
            raise ValueError("split must be 'train' or 'test'")
        
    def __getitem__(self, index):
        if isinstance(index, int):
            # If index is an int, convert to DataFrame index to handle non-sequential indices after split
            index = self.annotations.index[index]
        
        ecg_path = os.path.join(self.root_dir, self.annotations.loc[index, 'filepath'])
        
        f = open(ecg_path, "rb")
        ecg = np.fromfile(f, dtype=np.int16)
        ecg = np.reshape(ecg, (8, 5000))
        ecg = np.vstack([ecg, ecg[1] - ecg[0]]) # ecg_dict['III'] = ecg_dict['II'] - ecg_dict['I']
        ecg = np.vstack([ecg, -0.5 * (ecg[0] + ecg[1])]) # ecg_dict['aVR'] = -0.5 * (ecg_dict['I'] + ecg_dict['II'])
        ecg = np.vstack([ecg, ecg[0] - 0.5 * ecg[1]]) # ecg_dict['aVL'] = ecg_dict['I'] - 0.5 * ecg_dict['II']
        ecg = np.vstack([ecg, ecg[1] - 0.5 * ecg[0]]) # ecg_dict['aVF'] = ecg_dict['II'] - 0.5 * ecg_dict['I']
        f.close()
        
        label = self.annotations.loc[index, 'label']
            
        return ecg, label
    
    def __len__(self):
        return len(self.annotations)