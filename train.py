import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from data import TwoMoonsDataset
from model import ResFFN

def accuracy(probs, y_gnd):
    """ Computes the accuracy of the model's predictions. 
    
    Parameters:
    -----------
    probs : torch.Tensor, shape [batch_size, num_classes]
        Softmax probabilities for each class assignment predicted by the model.
    y_gnd : torch.Tensor, shape [batch_size]
        Ground-truth class label.
        
    Returns:
    --------
    accuracy : float
        Fraction of correctly predicted samples.
    """
    return (probs.argmax(1) == y_gnd).sum().item() / probs.shape[0]

def evaluate(model, data_loader, criterion):
    """ Evaluates loss and accuracy of model on a dataset (validation or test).
    
    Parameters:
    -----------
    model : torch.nn.Module
        A trained model.
    data_loader : torch.utils.data.DataLoader
        A data loader for the dataset to evaluate the model against.
    criterion : function
        A differentiable loss function.

    Returns:
    --------
    loss : float
        Average loss of samples in the dataset.
    accuracy : float
        Average accuracy of samples in the dataset.
    """
    model.eval()
    total_loss, num_correct = 0.0, 0

    with torch.no_grad():
        for x, y in data_loader:
            if torch.cuda.is_available():
                x, y = x.float().cuda(), y.cuda()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            num_correct += (y_pred.argmax(1) == y).sum().item()

    return total_loss / len(data_loader), num_correct / len(data_loader.dataset)


def train_model(model, data_train, data_val, epochs=150, batch_size=128):
    """ Trains the model.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to train.
    data_train : torch.utils.data.Dataset
        Training data.
    data_val : torch.utils.data.Dataset
        Validation data.
    epochs : int
        For how many epochs to train.
    batch_size : int
        The batch size during training.
        
    Returns:
    --------
    loss_history : np.array, shape [num_epochs]
        Average validation loss after each epoch.
    accuarcy_history : np.array, shape [num_epochs]
        Validation accuracy after each epoch.
    """
    data_loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=False)
    data_loader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    loss_history, accuracy_history = [], []
    for epoch in range(epochs):

        # Training
        print(f'### Epoch {epoch + 1} / {epochs}')
        running_loss, running_accuracy = 0.0, 0.0
        model.train()
        for batch_idx, (x, y) in enumerate(data_loader_train):
            if torch.cuda.is_available():
                x, y = x.float().cuda(), y.cuda()
            optimizer.zero_grad() 
            y_pred = model(x)
            loss = criterion(y_pred, y)
            running_loss += loss.item()
            running_accuracy += accuracy(y_pred, y)

            loss.backward()
            optimizer.step()

            #print(f'Batch {batch_idx + 1}. Running loss: {running_loss / (batch_idx + 1):.4f}; Running accuracy {running_accuracy / (batch_idx + 1):.4f}\r', end='\r')
        
        # Validation
        val_loss, val_accuracy = evaluate(model, data_loader_val, criterion)
        print(f'Validation loss {val_loss:.4f}; Validation accuracy {val_accuracy:.4f}')
        loss_history.append(val_loss)
        accuracy_history.append(val_accuracy)

    return np.array(loss_history), np.array(accuracy_history)

