from model.lstm_model import LSTM
from utils.access_log_dataset import AccessLogsDataset
from utils.config_loader import load_config
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm

def train():
    """
    Method to train the LSTM model.
    :return:
    """
    # load data and training configurations
    config = load_config()
    data_config = config['data']
    training_config = config['training']

    # load the dataset
    dataset = AccessLogsDataset(data_config['static_dataset_path'])
    dataloader = DataLoader(dataset=dataset, batch_size=training_config['batch_size'], shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # use LSTM customized model
    model = LSTM().to(device)

    # definition of the loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer to accelerate the learning process
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])

    for epoch in range(training_config['epochs']):

        # initialize the total loss to 0
        total_loss = 0.0

        # show the progress bar of the training process
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        # train the model
        for x, y in progress_bar:

            x, y = x.to(device), y.to(device)

            # forward pass
            outputs = model(x)

            # calculate the loss
            loss = criterion(outputs, y)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the total loss
            total_loss += loss.item()

            # update the progress bar showing the loss
            progress_bar.set_postfix(loss=loss.item())

    # save the trained model
    torch.save(model.state_dict(), training_config['model_save_path'])

train()
