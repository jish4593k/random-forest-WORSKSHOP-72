import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from matplotlib.colors import ListedColormap

# Importing the dataset
datasets = pd.read_csv('Social_Network_Ads.csv')
X = datasets.iloc[:, [2, 3]].values
Y = datasets.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Fitting the Random Forest Classifier into the Training set
classifier = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
classifier.fit(X_Train, Y_Train)

# Predicting the Test set results
Y_Pred = classifier.predict(X_Test)

# Making the Confusion Matrix
cm = confusion_matrix(Y_Test, Y_Pred)
print("Confusion Matrix:\n", cm)

# Print Classification Report
print("Classification Report:\n", classification_report(Y_Test, Y_Pred))

# Visualizing the Confusion Matrix with Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Purchased', 'Purchased'],
            yticklabels=['Not Purchased', 'Purchased'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualizing the Decision Boundary with Keras
model_keras = Sequential([
    Dense(units=6, activation='relu', input_dim=2),
    Dense(units=6, activation='relu'),
    Dense(units=1, activation='sigmoid')
])
model_keras.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the Keras Model
model_keras.fit(X_Train, Y_Train, epochs=50, batch_size=10)

# Visualizing the Decision Boundary with PyTorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model_pytorch = Net()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model_pytorch.parameters(), lr=0.01)

# Converting numpy arrays to PyTorch tensors
X_Train_tensor = torch.Tensor(X_Train)
Y_Train_tensor = torch.Tensor(Y_Train).view(-1, 1)

# Creating DataLoader for PyTorch
train_dataset = TensorDataset(X_Train_tensor, Y_Train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)

# Training the PyTorch Model
for epoch in range(50):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model_pytorch(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Visualizing the Decision Boundary
plt.figure(figsize=(12, 5))

# Visualizing with Keras
plt.subplot(1, 2, 1)
plt.contourf(X1, X2, np.array(model_keras.predict(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Decision Boundary with Keras')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

# Visualizing with PyTorch
plt.subplot(1, 2, 2)
X1_tensor = torch.Tensor(X1)
X2_tensor = torch.Tensor(X2)
plt.contourf(X1, X2, np.array(model_pytorch(X1_tensor)).detach().numpy().reshape(X1.shape),
             alpha=0.75, cmap=Listed
