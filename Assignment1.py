import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
# Device-configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size =1024
learning_rate = 0.001

# Dataset
train_dataset = torchvision.datasets.MNIST(root='data', 
                                           train=True, # training data part 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='data',
                                          train=False, download=False,
                                          transform=transforms.ToTensor())

#Dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self, aa, hidden_size, num_classes):
        super(NeuralNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=aa, out_channels=64, kernel_size=(3,3)) # convolution needs channels
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3) )
        self.max = nn.MaxPool2d((3,3), stride=(2,2))

        self.dro = nn.Dropout(p=0.2)


        # self.normal = nn.LayerNorm()
        self.fc1 = nn.Linear(in_features=128*11*11, out_features=hidden_size)#???? 64*12*12 dalta warta mung shape mention ku che da shape ba v. kho da ba pa mung aw malumaw print('--------', out4.shape).
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=num_classes) # fully connected need pixels
        self.act = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.act(out1)
        out77 = self.conv2(out2)
        out3 = self.max(out77)
        out4 = self.dro(out3)
        # out5 = self.normal(out4)


        # print('--------', out4.shape)
        out5 = torch.flatten(out4, start_dim=1)
        out6 = self.fc1(out5)
        out7 = self.act(out6)
        out8 = self.fc2(out7)
        out9 = self.act(out8)


        return out9
    

model = NeuralNet(1,500,10).to(device) # feed-forward---> input, hidden, output.

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


total_step = len(train_loader)

epoch_accuracy = []
train_accuracy = 0



for epoch in range(num_epochs):
    correct = 0
    total = 0
    for i, (images, labels) in (enumerate(tqdm(train_loader))):

        #images = torch.reshape(images, (10,784)).to(device)

        labels = labels.to(device)

        #print('images: ', images.shape)# to check the input shape and the number of channels

        images = images.to(device) #????

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### train accuracies
        out_probs = torch.softmax(outputs, dim=1)
        __, predicted = torch.max(out_probs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i+1) % 10 == 0: # ?????
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    
    epoch_accuracy.append(100 * correct / total)

for epoch in range(num_epochs):
    print('Accuracy of the network on the train images: {:.2f} %, on epoch {}'.format(epoch_accuracy[epoch], epoch+1))




with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        # images = torch.reshape(images, (10,784)).to(device)
        labels = labels.to(device)
        images = images.to(device)
        outputs = model(images)
        out_probs = torch.softmax(outputs, dim=1)
        __, predicted = torch.max(out_probs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))


torch.save(model.state_dict(), 'model.ckpt')
        






