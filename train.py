import torch
import torch.optim as optim
import torch.nn as nn
import model
from torch.utils.data import DataLoader
from torchvision import  transforms
from custom_MOT import custom_MOT
from tqdm import tqdm
import cv2

# learning parameters
learning_rate = 0.001
epochs = 10
batch_size = 4
#computation device
#device =  torch.device('cuda' if torch.cuda.is_available else 'cpu')

# train and validation datasets
train_data = custom_MOT(csv_file='data/MOT15/MOT_labels.csv', root_dir='data/MOT15/train/ADL-Rundle-6/img1')
val_data = custom_MOT(csv_file='data/MOT15/MOT_labels.csv', root_dir='data/MOT15/train/ADL-Rundle-6/img1')
#test_data = custom_MOT(csv_file='data/MOT15/MOT_labels.csv', root_dir='data/MOT15/train/TUD-Campus/img1')

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
    shuffle=True
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
        #train_running_correct += abs(torch.sub(target, outputs.data))
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss/len(dataloader.dataset)
    train_accuracy = train_running_correct/len(dataloader.dataset)
    return train_loss, train_accuracy

# validation function
def validate(model, dataloader, optimizer, criterion, val_data, epoch):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            data, target = data[0], data[1]
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # display original warped and network corrected images
            if epoch == 1:
                img = data[0].numpy()
                img = img.transpose(1, 2, 0)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                print(target[0].numpy())
                print('\n', outputs[0].numpy())
                tar = outputs[0].numpy()
                img_warp = cv2.warpPerspective(img, tar, (256, 256), borderMode=cv2.BORDER_CONSTANT, borderValue=(127,127,127))
                #img_warp = img_warp[0:256, 64:192]
                cv2.imshow('orig warped', img)
                cv2.imshow('new warped', img_warp)
                cv2.waitKey(0)

            val_running_loss += loss.item()
            #val_running_correct += abs(torch.sub(target, outputs.data))
        val_loss = val_running_loss/len(dataloader.dataset)
        val_accuracy = val_running_correct/len(dataloader.dataset)
        return val_loss, val_accuracy

# train for certain epochs
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_accuracy = fit(model, train_loader, 
                                                 optimizer, criterion, 
                                                 train_data)
    val_epoch_loss, val_epoch_accuracy = validate(model, val_loader, 
                                                 optimizer, criterion, 
                                                 val_data, epoch + 1)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Validation Loss: {val_epoch_loss:.4f}")