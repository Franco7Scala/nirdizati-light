#$from torchvision import datasets, transforms
import torch
from base import BaseDataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


"""class MnistDataLoader(BaseDataLoader):
    # MNIST data loading demo using BaseDataLoader 
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)"""


class TabularDataset(Dataset):
    def __init__(self, dataset_path=None, dataset_file=None, target_col='Label', header_in_data=True, training=True, scale = True):
        if training:
            self.x, self.y, _, _ = self.__read_files(dataset_path, dataset_file, target_col, header_in_data, scale)
        else:
            _, _, self.x, self.y = self.__read_files(dataset_path, dataset_file, target_col, header_in_data, scale)
        
    def __read_files(self, dataset_path, dataset_file,target_col='Label', header_in_data=True, scale=True):
        """ Load dataframes for training and testing 
            splits them into data and labels 
        
        Args:
        dataset_path : name of the dataset and the folder containing it in data directory
        dataset_name : prefix datasets file names
        target_col: target column name
        header_in_data: data headers
        scale: If true we apply StandardScaler() to fatures, we do not scale the target

        Returns:
         train_x, test_x: data in the form numpy arrays
         train_y, test_y: labels in the form numpy arrays
        """

        file_name_train = 'data/'+ dataset_path + '/' + dataset_file + '_train.csv.zip'
        file_name_test = 'data/'+ dataset_path + '/' + dataset_file + '_test.csv.zip'
         
        train_x = pd.read_csv(file_name_train, sep=',', header=0 if header_in_data else None)
        train_y = train_x[target_col].to_numpy(dtype=np.float32)
        train_x.drop(target_col, axis=1, inplace= True)
        train_x = train_x.to_numpy(dtype=np.float32)

        test_x = pd.read_csv(file_name_test, sep=',', header=0 if header_in_data else None)
        test_y = test_x[target_col].to_numpy(dtype=np.float32)
        test_x.drop(target_col, axis=1, inplace= True)
        test_x = test_x.to_numpy(dtype=np.float32)

        if scale: 
            scaler = StandardScaler()
            scaler.fit(train_x)
            train_x = scaler.transform(train_x)
            test_x = scaler.transform(test_x)

        return train_x, train_y, test_x, test_y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        features = torch.tensor(self.x[idx])
        label = torch.tensor([self.y[idx]])
        return features, label

class TabularDataLoader(BaseDataLoader):
    """
      #MNIST data loading demo using BaseDataLoader 
    """
    def __init__(self, dataset_path=None, dataset_file=None, target_col='Label', header_in_data=True, scale=True ,batch_size=8, shuffle=True, validation_split=0.0, num_workers=0, training=True):
        dataset = TabularDataset(dataset_path, dataset_file, target_col, header_in_data, training, scale)
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)
