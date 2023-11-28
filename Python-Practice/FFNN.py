import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784
hidden_size = 500 #neurons
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001


# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='data', 
                                           train=True, # training data part 
                                           transform=transforms.ToTensor(),  
                                           download=False)

test_dataset = torchvision.datasets.MNIST(root='data', 
                                          train=False, download=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Fully connected neural network with one hidden layer
# Define the Class
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, num_classes)  
        self.relu = nn.ReLU()

        # self.block = nn.sequential(
        #     nn.Linear(input_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, num_classes)

        # )
    
    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.relu(out1)
        out3 = self.fc2(out2)
       # out = self.block(x)
        return out3

model = NeuralNet(input_size, hidden_size, num_classes).to(device) #to(device)????????

# below process is same for all

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  



# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        #images = images.reshape(-1, 1*28*28).to(device) ## reshape the input to [B, ...]
        images = torch.reshape(images, (100,784)).to(device) ## reshape the input to [B, ...] we can also use 28 * 28 instead 784
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images) #forward pass
        loss = criterion(outputs, labels) # find the loss between output and label.
        
        # Backward and optimize
        optimizer.zero_grad() ## initialize the optimizer 
        loss.backward() ## weight calculation 
        optimizer.step() ## weight updates
        
        if (i+1) % 50 == 0:  ## loss to be shown after every 50th iteration........ this part is necessary. not related with training.
            out_probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(out_probs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item(), 100 * correct / total))


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        #images = images.reshape(-1, 28*28).to(device)
        images = torch.reshape(images, (100,784)).to(device) ## reshape the input to [B, ...]
        labels = labels.to(device)
        outputs = model(images)
        out_probs = torch.softmax(outputs, dim=1) # output shape is (100, 10) 100 is on 0 dim and 10 on 1st dimension 
        _, predicted = torch.max(out_probs.data, 1)

        total += labels.size(0) # this formula is set manually. we can also take from internet
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')