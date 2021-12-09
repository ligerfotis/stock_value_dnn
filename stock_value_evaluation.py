import numpy as np
import torch

from DataLoader import FinancialDataLoader
def evaluation():
    model_dir = "model_dir_512_512/"
    model_name = "new_model"

    financial_loader = FinancialDataLoader()
    train_loader, test_loader = financial_loader.prepare_financial_dataset(10)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    new_model = torch.load(model_dir + model_name)
    samples_x = np.array(financial_loader.test_x)
    samples_y = np.array(financial_loader.test_y)

    for i in range(10):
        prediction = new_model(torch.Tensor(samples_x[i]).to(device)).item()
        print("Prediction {2:} - Real Value: {0:.2f}-{1:.2f}".format(prediction, samples_y[i][0], i + 1))
