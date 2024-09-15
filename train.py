import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train_step( model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> Tuple[float, float]:
    
    '''
    Trains a pytorch model for a single epoch
    '''

    model.train()
    train_loss , train_acc = 0, 0

    for batch, (X,y) in enummerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                device: torch.device):
    '''
    Test step for a single epoch
    '''
    model.eval()

    test_loss , test_acc = 0,0
    with torch.inference_mode():
        for batch, (X,y) in ennumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)

            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            train_acc += (test_pred_class == y).sum().item()/len(test_pred_labels)
    
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)



def train(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                epochs: int,
                device = torch.device):

    '''
    Training and testing a pytorch model for classification
    '''
    
    results = { 
        train_acc = [],
        train_loss = [],
        test_acc =[],
        test_loss = []
    }







