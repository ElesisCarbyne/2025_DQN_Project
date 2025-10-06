import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding) # 8 x 8 x 32 or 4 x 4 x 64 || 6 x 6 x 32 or 3 x 3 x 64
        self.batch_norm2d_1 = nn.BatchNorm2d(num_features=out_channels) # 8 x 8 x 32 or 4 x 4 x 64 || 6 x 6 x 32 or 3 x 3 x 64
        self.relu = nn.ReLU() # 8 x 8 x 32 or 4 x 4 x 64 || 6 x 6 x 32 or 3 x 3 x 64
        self.conv2d_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1) # 8 x 8 x 32 or 4 x 4 x 64 || 6 x 6 x 32 or 3 x 3 x 64
        self.batch_norm2d_2 = nn.BatchNorm2d(num_features=out_channels) # 8 x 8 x 32 or 4 x 4 x 64 || 6 x 6 x 32 or 3 x 3 x 64

    def forward(self, x):
        result = self.conv2d_1(x)
        result = self.batch_norm2d_1(result)
        result = self.relu(result)
        result = self.conv2d_2(result)
        result = self.batch_norm2d_2(result)
        return result

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv_block1 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_block2 = ConvBlock(out_channels, out_channels)
        self.relu = nn.ReLU()
        
        self.in_out_channels_same = True if in_channels == out_channels else False
        if not self.in_out_channels_same:
            self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        result = self.relu(self.conv_block1(x) + self.conv2d(x)) if not self.in_out_channels_same else self.relu(self.conv_block1(x) + x)
        return self.relu(self.conv_block2(result) + result)

class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        result = self.linear(x)
        result = self.relu(result)
        result = self.dropout(result)
        return result

class FFNBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(FFNBlock, self).__init__()
        self.flatten = nn.Flatten()
        self.dense_block1 = DenseBlock(in_features, out_features[0])
        self.dense_block2 = DenseBlock(out_features[0], out_features[0])
        self.dense_block3 = DenseBlock(out_features[0], out_features[0])
        self.dense_block4 = DenseBlock(out_features[0], out_features[1])
        self.linear = nn.Linear(in_features=out_features[1], out_features=4)

    def forward(self, x):
        result = self.dense_block1(self.flatten(x))
        result = self.dense_block2(result)
        result = self.dense_block3(result)
        result = self.dense_block4(result)
        return self.linear(result)
        
class Agent(nn.Module):
    def __init__(self, input_size):
        super(Agent, self).__init__()
        self.blockA = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=6, stride=2, padding=2), # 16 x 16 x 32 || 12 x 12 x 32
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1) # 8 x 8 x 32 || 6 x 6 x 32
        )
        self.blockB = ResidualBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1) # 8 x 8 x 32 || 6 x 6 x 32
        match input_size:
            case 32:
                self.blockC = ResidualBlock(in_channels=32, out_channels=64, kernel_size=3, stride=3, padding=2) # 4 x 4 x 64
                self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0) # 1 x 1 x 64
            case 24:
                self.blockC = ResidualBlock(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1) # 3 x 3 x 64
                self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=0) # 1 x 1 x 64
        self.ffn = FFNBlock(in_features=64, out_features=[512, 256])

    def forward(self, x):
        result = self.blockA(x)
        result = self.blockB(result)
        result = self.blockC(result)
        result = self.avg_pool(result)
        return self.ffn(result)

if __name__ == "__main__":
    from torchsummary import summary
    
    model = Agent(input_size=24)
    summary(model, (3, 24, 24))