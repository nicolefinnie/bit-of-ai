import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """Just a simple CNN suitable for CIFAR-10 (32x32) for demonstrating fusion of conv2+bn2
    
    input shape = (B, 3, 32, 32)
    output feature = (input feature + 2*padding - kernel_size) / stride + 1
    
    1. so after the first conv layer [B, 32, 32, 32]
    (32 + 2*1 - 3 )/1 + 1 = 32
    and we pool it to reduce the feature dimension by half [B, 32, 16, 16]

    2. after the 2nd conv layer
    (16 + 2*1 - 3 )/1 + 1 = 16
    after the second conv layer [B, 64, 16, 16]
    we pool it to reduce the feature dimension by half [B, 64, 8, 8]

    3. The linear layers are straightforward, we reduce the spatial dimensions from [B, 64, 8, 8] 
        to [B, 64*8*8] to [B, 128] to [B, 64] to [B, 10] because CIFAR-10 predicts 10 classes

    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(64 * 8 * 8, 128) # 3 pooling layers reduce spatial dimensions to 4x4
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        

    def forward(self, x):
        # pooling layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)

        # pooling layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)

        # flatten pooled conv features to fully connected layers
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return x