import torch
import torch.nn as nn


class CNN_v0(nn.Module):
    def __init__(self, action_dim):
        super(CNN_v0, self).__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), 
            nn.ReLU()
        )
        
        self.convblock2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), 
            nn.ReLU()
        )
        
        self.convblock3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), 
            nn.ReLU()
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv1d(128, 50, kernel_size=3, padding=1), 
            nn.BatchNorm1d(50), 
            nn.Flatten(), 
            nn.Linear(50 * 249, action_dim) # 4 * action_dim
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv1d(128, 50, kernel_size=3, padding=1), 
            nn.BatchNorm1d(50), 
            nn.Flatten(), 
            nn.Linear(50 * 249, 128), 
            nn.Linear(128, 1), 
            # nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
    
    
class CNN_v1(nn.Module):
    def __init__(self, action_dim, num_resBlocks=8):
        super(CNN_v1, self).__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), 
            nn.ReLU()
        )
        
        self.convblock2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), 
            nn.ReLU()
        )
        
        self.convblock3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), 
            nn.ReLU()
        )
        
        self.resblocks = nn.ModuleList(
            [ResBlock(128) for i in range(num_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv1d(128, 50, kernel_size=3, padding=1), 
            nn.BatchNorm1d(50), 
            nn.Flatten(), 
            nn.Linear(50 * 249, action_dim) # 4 * action_dim
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv1d(128, 50, kernel_size=3, padding=1), 
            nn.BatchNorm1d(50), 
            nn.Flatten(), 
            nn.Linear(50 * 249, 128), 
            nn.Linear(128, 1), 
            # nn.Tanh()
        )
    
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        
        for resBlock in self.resblocks:
            x = resBlock(x)
        
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv1d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_hidden)
        self.conv2 = nn.Conv1d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_hidden)
        
        self.activation = nn.ReLU()
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.activation(x)
        
        return x