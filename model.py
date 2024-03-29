import torch
import torch.nn as nn

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        # spatial transformer localisation network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # tranformation regressor for theta
        self.fc_loc = nn.Sequential(
            nn.Linear(128*60*60, 256),
            nn.ReLU(True),
            nn.Linear(256, 3 * 3)
        )
        # initializing the weights and biases with identity transformations
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1],
                                                    dtype=torch.float))
    

    def forward(self, x):
        xs = self.localization(x)                           # get feature map from localiser
        xs = xs.view(-1, xs.size(1)*xs.size(2)*xs.size(3))  # resize feature map
        theta = self.fc_loc(xs)                             # calculate the transformation parameters theta
        theta = theta.view(-1, 3, 3)                        # resize theta into 3x3 transformation matrix
        return theta
