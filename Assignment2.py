import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784
hidden_size = 500
num_classes = 10
batch_size = 1024
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root='data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='data', 
                                          train=False, download=False,
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self, aa,  num_classes):
        super(NeuralNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=aa, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.max1 = nn.MaxPool2d((2,2), stride=(2,2))
        self.max2 = nn.MaxPool2d((2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.act = nn.ReLU()

        self.fc1 = nn.Linear(in_features=64*7*7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=num_classes)


    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.act(out1)
        out3 = self.max1(out2)
        out4 = self.conv2(out3)
        out5 = self.act(out4)
        out6 = self.max2(out5)

        out7 = torch.flatten(out6, start_dim=1)
        out8 = self.fc1(out7)
        out9 = self.act(out8)
        out10 = self.fc2(out9)
        out11 = self.act(out10)
        out12 = self.fc3(out11)
        out13 = self.act(out12)
        out14 = self.fc4(out13)

        return out14
        

        # print(out6.shape)
        


model = NeuralNet(1,10).to(device) # feed-forward---> input, hidden, output.

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
num_epochs = 20

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

        optimizer.zero_grad() # initialize the optimizer
        loss.backward() # weight calculation
        optimizer.step() # weight updates
        

        ### train accuracies
        out_probs = torch.softmax(outputs, dim=1)
        __, predicted = torch.max(out_probs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i+1) % 10 == 0: # ?????
            print ('\nEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    
    epoch_accuracy.append(100 * correct / total)

for epoch in range(num_epochs):
    print('Accuracy of the network on the train images: {:.2f} %, on epoch {}'.format(epoch_accuracy[epoch], epoch+1))



correct = 0
total = 0
for i, (images, labels) in (enumerate(tqdm(test_loader))):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)


    out_probs = torch.softmax(outputs, dim=1) ### convert into range of [0 1]
    __, predicted = torch.max(out_probs.data, 1) ## only take the maximum class probability

    total += labels.size(0) #total no. of samples in all iterations 
    correct += (predicted == labels).sum().item() ### total no. of correct predictions in all iterations

print("Test accuracy {:.2f}".format(correct/total * 100))


  


