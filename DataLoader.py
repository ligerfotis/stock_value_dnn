import pandas as pd
import torch
import numpy as np


class FinancialDataLoader:
    def __init__(self):
        self.train_samples = None
        self.test_x = None
        self.test_y = None
        self.train_x = None
        self.train_y = None

    def prepare_financial_dataset(self, batch_size):
        financial = pd.read_csv('data/financials.csv')
        # print(financial.columns)

        input_names = ['Sector', 'Price/Earnings', 'Dividend Yield',
                       'Earnings/Share', '52 Week Low', '52 Week High', 'Market Cap', 'EBITDA',
                       'Price/Sales', 'Price/Book', 'Price']
        label_name = 'Price'
        # drop nan rows
        financial.dropna(inplace=True)
        for col in financial.columns:
            if col not in input_names:
                del financial[col]
        for col in input_names:
            if financial[col].dtypes not in ("float64", "int64"):
                keys = financial[col].unique()
                categorical_values, categorical_keys = pd.factorize(keys)
                for key, value in zip(categorical_keys, categorical_values):
                    financial.replace(to_replace=key, value=value, inplace=True)
                print("column {} factorized to {}".format(col, financial[col].dtypes))

        # split the dataset to input and labels
        labels = pd.DataFrame(financial[label_name])
        del financial[label_name]
        # normalize features
        for col in financial.columns:
            financial[col] = (financial[col] - financial[col].min()) / (financial[col].max() - financial[col].min())
            # financial[col] = (financial[col] - financial[col].mean()) / financial[col].std()

        # split into train(80) and test(20) sets
        self.train_x = financial.iloc[:self.train_samples, :]
        self.train_y = labels.iloc[:self.train_samples, :]
        self.test_x = financial.iloc[self.train_samples:, :]
        self.test_y = labels.iloc[self.train_samples:, :]

        print("Dataset: {} Features {} Train Samples {} Test Samples".format(self.train_x.shape[1], self.train_x.shape[0], self.test_x.shape[1]))

        # convert to Dataset
        train_dataset = torch.utils.data.TensorDataset(torch.Tensor(np.array(self.train_x)),
                                                       torch.Tensor(np.array(self.train_y)))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_dataset = torch.utils.data.TensorDataset(torch.Tensor(np.array(self.test_x)),
                                                      torch.Tensor(np.array(self.test_y)))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
