import torch
import torch.optim as optim
import torch.nn as nn
import model
from torch.utils.data import DataLoader
from custom_MOT import custom_MOT
import cv2
import numpy as np


SHOW_VAL = True     # whether to display images when in validation loop

# dataset names and directories
DATA_DIR = 'data/MOT15/train/'
CSV_DIR = 'data/MOT15/'

TRAIN_SET = 'ADL-Rundle-6'
VAL_SET = TRAIN_SET
TEST_SET = 'TUD-Campus'


TRAIN_DIR = DATA_DIR + TRAIN_SET + '/img1'
TRAIN_CSV = CSV_DIR + 'train_data.csv'

VAL_DIR = DATA_DIR + VAL_SET + '/img1'
VAL_CSV = CSV_DIR + 'val_data.csv'

TEST_DIR = DATA_DIR + TEST_SET + '/img1'
TEST_CSV = CSV_DIR + 'test_data.csv'


# learning parameters
learning_rate = 0.00001
epochs = 20
batch_size = 16

#computation device
#device =  torch.device('cuda' if torch.cuda.is_available else 'cpu')

# train and validation datasets
train_data = custom_MOT(csv_file=TRAIN_CSV, root_dir=TRAIN_DIR)
val_data = custom_MOT(csv_file=VAL_CSV, root_dir=VAL_DIR)
test_data = custom_MOT(csv_file=TEST_CSV, root_dir=TEST_DIR)

# train data loader
train_loader = DataLoader(
    train_data, 
    batch_size=batch_size,
    shuffle=True,
)
# val data loader
val_loader = DataLoader(
    val_data, 
    batch_size=batch_size,
    shuffle=True
)
# test data loader
test_loader = DataLoader(
    test_data, 
    batch_size=batch_size,
    shuffle=True
)

model = model.STN()                                             # model
optimizer = optim.Adam(model.parameters(), lr=learning_rate)    # optimizer
criterion = nn.MSELoss()                                        # loss function


def display(data, output, target, last):
    """Display original, network transformed and target transformed images in validation loop"""
    img = data[0].numpy()
    img = img.transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
    print(np.around(target[0].numpy(), 3))
    print('\n', np.around(output[0].numpy(), 4))

    img_warp = cv2.warpPerspective(img, output[0].numpy(), (256, 256), borderMode=cv2.BORDER_CONSTANT, borderValue=(127,127,127))
    img_warp_ideal = cv2.warpPerspective(img, target[0].numpy(), (256, 256), borderMode=cv2.BORDER_CONSTANT, borderValue=(127,127,127))

    cv2.imshow('original', img)
    cv2.imshow('network out', img_warp)
    cv2.imshow('target out', img_warp_ideal)
    cv2.waitKey(0)
    if last:
        cv2.destroyAllWindows()


# training function
def train(model, dataloader, optimizer, criterion, train_data):
    """Training loop"""
    print('Training\n')
    model.train()
    train_running_loss = 0.0
    for i, data in enumerate(dataloader):
        print('Batch {}/{}'.format(i, int(len(train_data)/dataloader.batch_size)))
        data, target = data[0], data[1]
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target.float())
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()

        print(f'Loss:\n{loss.item():.3f}')
        print('Target:\n', np.around(target[0].detach().numpy(), 3))
        print('Output:\n', np.around(outputs[0].detach().numpy(), 4), '\n')
    train_loss = train_running_loss/len(dataloader.dataset)
    return train_loss

# validation function
def validate(model, dataloader, optimizer, criterion, val_data):
    """Validation loop"""
    print('Validating\n')
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            last = False
            print('Batch {}/{}'.format(i, int(len(val_data)/dataloader.batch_size)))
            data, target = data[0], data[1]
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # display original, network and target corrected images
            if SHOW_VAL:
                if i == int(len(val_data)/dataloader.batch_size):
                    last = True
                display(data, outputs, target, last)

            val_running_loss += loss.item()
        val_loss = val_running_loss/len(dataloader.dataset)
        return val_loss

# train for num of epochs
for epoch in range(epochs):
    train_epoch_loss = train(model, train_loader, optimizer, criterion, train_data)
    val_epoch_loss = validate(model, val_loader, optimizer, criterion, val_data)

    print(f"\nEpoch {epoch+1} of {epochs}")
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Validation Loss: {val_epoch_loss:.4f}")
