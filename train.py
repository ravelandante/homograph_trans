import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import model
from torch.utils.data import DataLoader
from torchvision import  transforms
from load_dataset.custom_MOT import custom_MOT
from tqdm import tqdm

# learning parameters
learning_rate = 0.001
epochs = 10
batch_size = 4
# image transforms
transform = transforms.Compose([
                       transforms.ToTensor(),
                   ])
#computation device
#device =  torch.device('cuda' if torch.cuda.is_available else 'cpu')

# train and validation datasets
train_data = custom_MOT(csv_file='load_dataset/MOT_data/MOT_labels.csv', root_dir='load_dataset/MOT_data/train/')
val_data = custom_MOT(csv_file='load_dataset/MOT_data/MOT_labels.csv', root_dir='load_dataset/MOT_data/train/')

# train data loader
train_loader = DataLoader(
    train_data, 
    batch_size=batch_size,
    shuffle=True
)
# train data loader
val_loader = DataLoader(
    val_data, 
    batch_size=batch_size,
    shuffle=False
)

# initialize the model
model = model.STN()
# initialize the optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# initilaize the loss function
criterion = nn.CrossEntropyLoss()

# training function
def fit(model, dataloader, optimizer, criterion, train_data):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data, target = data[0], data[1]
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss/len(dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(dataloader.dataset)    
    return train_loss, train_accuracy

# validation function
def validate(model, dataloader, optimizer, criterion, val_data):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            data, target = data[0], data[1]
            outputs = model(data)
            loss = criterion(outputs, target)
            
            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()
        
        val_loss = val_running_loss/len(dataloader.dataset)
        val_accuracy = 100. * val_running_correct/len(dataloader.dataset)        
        return val_loss, val_accuracy

# train for certain epochs
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_accuracy = fit(model, train_loader, 
                                                 optimizer, criterion, 
                                                 train_data)
    val_epoch_loss, val_epoch_accuracy = validate(model, val_loader, 
                                                 optimizer, criterion, 
                                                 val_data)
    print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}")
    print(f"Validation Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}")