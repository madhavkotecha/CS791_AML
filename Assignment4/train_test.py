### Add necessary imports ###
from torch.utils.data import DataLoader
import torch
from models import SimpleNN, CNN

def train_and_test_NN(datasets, hyperparams, seed=42):
    """
    Train and test a Neural Network model
    datasets: tuple of (train_dataset, test_dataset)
    hyperparams: data structure containing hyperparameters like learning rate, epochs, etc.

    Returns:
    accuracy: accuracy on validation dataset
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset, validation_dataset = datasets

    # STEP 1: Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=hyperparams["batch_size"],  shuffle=False)

    # STEP 2: Initialize model, optimizer, CE loss
    model = SimpleNN(hidden_size=hyperparams["hidden_size"], dropout_rate=hyperparams["dropout_rate"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
    loss = torch.nn.CrossEntropyLoss()

    ### Implement training loop here ###
    # STEP 3: Train the model for the specified number of epochs.
    accuracy = None
    model.train()
    num_epochs = hyperparams["training_epochs"]
    for ep in range(num_epochs):
        epoch_loss = 0.0
        for imgs, labels in train_dataloader:
            optimizer.zero_grad()
            imgs =  imgs.to(device)
            labels = labels.to(device)
            out_classes = model(imgs)
            train_loss = loss(out_classes, labels)
            train_loss.backward()
            optimizer.step()
            epoch_loss += train_loss.item()
        print(f"Epoch {ep+1}: Train loss: {epoch_loss/len(train_dataloader)}")
    
    # STEP 4: Evaluate
    ### Implement validation loop here ###
    model.eval()
    total_correct_predictions = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_dataloader:
            imgs =  imgs.to(device)
            labels = labels.to(device)
            out_classes = model(imgs)
            _, predictions =  torch.max(out_classes, dim = -1)
            correct = (predictions == labels).sum()
            total_correct_predictions += correct.item()
            total += labels.shape[-1]
    accuracy = 100.0 * total_correct_predictions / total
    
    return accuracy

def train_and_test_CNN(datasets, hyperparams, seed=42):
    """
    Train and test a Convolutional Neural Network model
    datasets: tuple of (train_dataset, test_dataset)
    hyperparams: data structure containing hyperparameters like learning rate, epochs, etc.

    Returns:
    accuracy: accuracy on validation dataset
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset, validation_dataset = datasets

    train_dataloader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=hyperparams["batch_size"],  shuffle=False)

    model = CNN(hidden_size=hyperparams["hidden_size"], dropout_rate=hyperparams["dropout_rate"]).to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=hyperparams["lr"], momentum=0.9)

    if hyperparams["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=hyperparams["lr"], alpha=0.9, eps=1e-08, weight_decay=hyperparams["weight_decay"])

    loss = torch.nn.CrossEntropyLoss()

    ### Implement training loop here ###
    accuracy = None 
    model.train()
    num_epochs = hyperparams["training_epochs"]
    for ep in range(num_epochs):
        epoch_loss = 0.0
        for imgs, labels in train_dataloader:
            imgs =  imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out_classes = model(imgs)
            train_loss = loss(out_classes, labels)
            train_loss.backward()
            optimizer.step()
            epoch_loss += train_loss.item()
        print(f"Epoch {ep+1}: Train loss: {epoch_loss/len(train_dataloader)}")

    
    ### Implement validation loop here ###

    model.eval()
    total_correct_predictions = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_dataloader:
            imgs =  imgs.to(device)
            labels = labels.to(device)
            out_classes = model(imgs)
            _, predictions =  torch.max(out_classes, dim = -1)
            correct = (predictions == labels).sum()
            total_correct_predictions += correct.item()
            total += labels.shape[-1]
    accuracy = 100.0 * total_correct_predictions / total
    
    return accuracy
