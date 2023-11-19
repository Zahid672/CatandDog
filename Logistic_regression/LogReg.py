from torch import nn

class LogReg(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LogReg, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes) 
        self.dp = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.con1 = nn.Conv2d()
    
    def forward(self, x):
        out = self.fc1(x)
        return out