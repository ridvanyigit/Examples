import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Loading the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Loading the training and test sets
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Calculating the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Converting the data into a user-row, movie-column format
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(ratings)
    return new_data

training_set = torch.FloatTensor(convert(training_set))
test_set = torch.FloatTensor(convert(test_set))

# Neural network architecture
class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

# Creating the model
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

# Training the model
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0
    for id_user in range(nb_users):
        input = training_set[id_user].unsqueeze(0)
        target = input.clone()

        if torch.sum(target > 0) > 0:
            output = sae(input)
            target[target == 0] = -1
            loss = criterion(output[target != -1], target[target != -1])
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            s += 1
    print(f'Epoch: {epoch}, Loss: {train_loss / s:.4f}')

# Testing the model
test_loss = 0
s = 0
for id_user in range(nb_users):
    input = test_set[id_user].unsqueeze(0)
    target = input.clone()

    if torch.sum(target > 0) > 0:
        output = sae(input)
        target[target == 0] = -1
        loss = criterion(output[target != -1], target[target != -1])
        test_loss += loss.item()
        s += 1

print(f'Test Loss: {test_loss / s:.4f}')

