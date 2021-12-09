from sys import maxsize
import numpy as np
import torch
from torch import nn

from DataLoader import FinancialDataLoader
from NN_models import NeuralNetwork
from utils import model_path_creator

# hyperparameters
save = True
model_name = "new_model_1e3lr"
learning_rate = 1e-3
batch_size = 128
epochs = 100000
l1_size = 512
l2_size = 512


def train_loop(dataloader, model, loss_fn, optimizer, device):
    # size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        prediction = model(X.to(device))
        loss = loss_fn(prediction, y.to(device))

        # Backpropagation
        optimizer.zero_grad()  # set optimizer's gradients to zero
        loss.backward()  # backpropagate the prediction loss by adjusting the gradients of the loss
        optimizer.step()  # adjust the parameters of the optimizer by the gradients collected in he backward pass

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
    return loss


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, avg_mse = 0, 0

    with torch.no_grad():  # avoid unnecessary gradient computations
        for X, y in dataloader:
            prediction = model(X.to(device))
            test_loss += loss_fn(prediction, y.to(device)).item()
    test_loss /= num_batches
    print(f"Test Avg loss: {test_loss:>8f} ")

    return test_loss


def fit():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create the NN model
    model = NeuralNetwork(input_dim=10, l1_size=l1_size, l2_size=l2_size, output_size=1).to(device)
    # create a dataloader for the financial dataset
    financial_loader = FinancialDataLoader()
    train_loader, test_loader = financial_loader.prepare_financial_dataset(batch_size)

    # Initialize the loss function
    loss_fn = nn.MSELoss()
    # Initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    min_test_loss = maxsize
    model_dir = "model_dir_{}_{}/".format(l1_size, l2_size)
    model_path_creator(model_dir)
    for t in range(epochs):
        loss = train_loop(train_loader, model, loss_fn, optimizer, device)
        if t % 2000 == 0:
            print(f"Epoch {t + 1} train loss: {loss:>7f}")
            test_loss = test_loop(test_loader, model, loss_fn, device)
            if save is True and test_loss < min_test_loss:
                min_test_loss = test_loss
                torch.save(model, model_dir + "new_model")
                print("Model saved")
    print("Training Done!")
