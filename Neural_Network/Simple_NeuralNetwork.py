import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

torch.manual_seed(41)
model = Model()

import pandas as pd
import matplotlib.pyplot as plt



url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
my_df = pd.read_csv(url)

print(my_df.tail())

my_df['variety'] = my_df['variety'].replace('Setosa', 0.0)
my_df['variety'] = my_df['variety'].replace('Versicolor', 1.0)
my_df['variety'] = my_df['variety'].replace('Virginica', 2.0)
print(my_df)

X = my_df.drop('variety', axis=1)
y = my_df['variety']

X = X.values
y = y.values

import sklearn
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(model.parameters)

# Train our model!
# Epochs? (one run thru all the training data in our network)
epochs = 100
losses = []
for i in range(epochs):
  # Go forward and get a prediction
  y_pred = model.forward(X_train) # Get predicted results

  # Measure the loss/error, gonna be high at first
  loss = criterion(y_pred, y_train) # predicted values vs the y_train

  # Keep Track of our losses
  losses.append(loss.detach().numpy())

  # print every 10 epoch
  if i % 10 == 0:
    print(f'Epoch: {i} and loss: {loss}')

  # Do some back propagation: take the error rate of forward propagation and feed it back
  # thru the network to fine tune the weights
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  # Graph it out!
plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel('Epoch')

with torch.no_grad():
   y_eval = model.forward(X_test)
   loss = criterion(y_eval, y_test)

   print(loss)

correct = 0
with torch.no_grad():
  for i, data in enumerate(X_test):
    y_val = model.forward(data)

    if y_test[i] == 0:
      x = "Setosa"
    elif y_test[i] == 1:
      x = 'Versicolor'
    else:
      x = 'Virginica'


    # Will tell us what type of flower class our network thinks it is
    print(f'{i+1}.)  {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')

    # Correct or not
    if y_val.argmax().item() == y_test[i]:
      correct +=1

print(f'We got {correct} correct!')

torch.save(model.state_dict(), 'my_really_awesome_iris_model.pt')

new_model = Model()
new_model.load_state_dict(torch.load('my_really_awesome_iris_model.pt'))

print(new_model.eval())


with torch.no_grad():
   y_eval = model.forward(X_test)
   loss = criterion(y_eval, y_test)

print(loss)

correct = 0
with torch.no_grad():
   for i, data in enumerate(X_test):
      y_val = model.forward(data)

      if y_test[i] == 0:
         x = "Setosa"
      elif y_test[i] == 1:
         x = 'Versicolor'
      else:
         x = 'Virginica'

print(f'{i+1}.) {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')

if y_val.argmax().item() == y_test[i]:
   correct +=1

print(f'We got {correct} correct!')

new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])

with torch.no_grad():
   print(model(new_iris))

newer_iris = torch.tensor([5.9, 3.0, 5.1, 1.8])

with torch.no_grad():
   print(model(newer_iris))

torch.save(model.state_dict(), 'my_really_awesome_iris_model.pt')

new_model = Model()
new_model.load_state_dict(torch.load('my_really_awesome_iris_model.pt'))

new_model.eval()