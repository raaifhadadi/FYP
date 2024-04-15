import torch
import pandas as pd
import os
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class PTBXL(Dataset):
    def __init__(self, root_dir, split='train', test_size=0.2, random_state=None, transform=None):
        
        self.root_dir = root_dir
        self.csv_file = os.path.join(root_dir, 'ptbxl_database.csv')
        self.scp_statements = os.path.join(root_dir, 'scp_statements.csv')
        
        # Load annotations and scp_statements.csv for diagnostic aggregation
        self.annotations = pd.read_csv(self.csv_file, index_col="ecg_id")
        self.agg_df = pd.read_csv(self.scp_statements, index_col=0)
        self.agg_df = self.agg_df[self.agg_df.diagnostic == 1]
        self.superclasses = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        
        # Apply diagnostic superclass to annotations
        self.annotations['diagnostic_superclass'] = [self.aggregate_diagnostic(ast.literal_eval(x), self.agg_df) for x in self.annotations.scp_codes]
        
        # Split the annotations DataFrame into train and test sets
        self.annotations_train, self.annotations_test = train_test_split(self.annotations, test_size=test_size, random_state=random_state)
        
        self.transform = transform
        
        if split == 'train':
            self.annotations = self.annotations_train
        elif split == 'test':
            self.annotations = self.annotations_test
        else:
            raise ValueError("split must be 'train' or 'test'")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            # If index is an int, convert to DataFrame index to handle non-sequential indices after split
            index = self.annotations.index[index]
        
        ecg_path = os.path.join(self.root_dir, self.annotations.loc[index, 'filename_hr'] + '.dat')
        
        with open(ecg_path, "rb") as f:
            ecg = np.fromfile(f, dtype=np.int16).reshape((5000, 12)).transpose()
        
        label = self.aggregate_diagnostic(ast.literal_eval(self.annotations.loc[index, 'scp_codes']), self.agg_df)
        label = [1 if superclass in label else 0 for superclass in self.superclasses]
        label = np.array(label)
            
        return ecg, label
    
    def get_codes_from_label(self, label):
        return [self.superclasses[i] for i in range(len(label)) if label[i] == 1]
    
    @staticmethod
    def aggregate_diagnostic(y_dic, agg_df):
        return list({agg_df.loc[key].diagnostic_class for key in y_dic if key in agg_df.index})