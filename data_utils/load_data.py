from torch.utils.data import Dataset, DataLoader

from data_utils.vocab import Vocab
import pandas as pd
import os

class MyDataset(Dataset):
    def __init__(self, data_path, vocab= None):
        super(MyDataset, self).__init__()
        data = pd.read_csv(data_path, encoding= 'utf-8')
        self.sentences = []
        self.sentiments = []

        self.vocab = vocab
        for i in range(len(data)):
            self.sentences.append(data.iloc[i]["sentence"])
            self.sentiments.append(data.iloc[i]["sentiment"])

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        return {
            "sentence": self.sentences[index],
            "label": self.sentiments[index]
        }
    
class Load_Data:
    def __init__(self, config):
        self.train_batch = config["train_batch"]
        self.dev_batch = config["dev_batch"]
        self.test_batch = config["test_batch"]

        self.dataset_folder = config['dataset']['dataset_folder']
        self.train_path = config['dataset']["train_path"]
        self.dev_path = config['dataset']["dev_path"]
        self.test_path = config['dataset']["test_path"]

    def load_train_dev(self):
        train_dataset = MyDataset(os.path.join(self.dataset_folder, self.train_path))
        dev_dataset = MyDataset(os.path.join(self.dataset_folder, self.dev_path))

        train_dataloader = DataLoader(train_dataset, self.train_batch, shuffle= True)
        dev_dataloader = DataLoader(dev_dataset, self.dev_batch, shuffle= False)

        return train_dataloader, dev_dataloader
    
    def load_test(self):
        test_dataset = MyDataset(os.path.join(self.dataset_folder, self.test_path))
        test_dataloader = DataLoader(test_dataset, self.test_batch, shuffle= False)

        return test_dataloader