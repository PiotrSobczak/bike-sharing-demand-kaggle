import numpy as np
from data_utils import DataUtils as du
import torch
from torch.autograd import Variable

TOTAL_DATASET_SIZE = 10887

dtype = torch.Tensor

def rmsle(y_pred,y_true):
    log_err = torch.log(y_pred + 1) - torch.log(y_true + 1)
    squared_le = torch.pow(log_err,2)
    mean_sle = torch.mean(squared_le)
    root_msle = torch.sqrt(mean_sle)
    return (root_msle)

datasetX,datasetY = du.get_processed_df('data/train.csv')

epochs = 50000
train_data_size = 9600
batch_size = 64
steps_in_epoch = train_data_size//batch_size
layer_dims = {"in": 13, "fc1": 10, "fc2": 10, "fc3": 10,"fc4": 10, "out": 1}

#Dividing data to train and val datasets, conversion from DF to numpyarray
X_train = np.array(datasetX[:train_data_size])
Y_train = np.array(datasetY[:train_data_size])
X_val = np.array(datasetX[train_data_size:])
Y_val = np.array(datasetY[train_data_size:])

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
X_train = Variable(torch.Tensor(X_train))
Y_train = Variable(torch.Tensor(Y_train))
X_train_batch = Variable(torch.randn(batch_size, 13))
Y_train_batch = Variable(torch.randn(batch_size))
X_val = Variable(torch.Tensor(X_val))
Y_val = Variable(torch.Tensor(Y_val))

# Use the nn package to define our model and loss dunction.
model = torch.nn.Sequential(
    torch.nn.Linear(layer_dims["in"], layer_dims["fc1"]),
    torch.nn.Tanh(),
    torch.nn.Linear(layer_dims['fc1'],layer_dims['fc2']),
    torch.nn.Tanh(),
    torch.nn.Linear(layer_dims['fc2'],layer_dims['fc3']),
    torch.nn.Tanh(),
    torch.nn.Linear(layer_dims['fc3'],layer_dims['fc4']),
    torch.nn.Tanh(),
    torch.nn.Linear(layer_dims['fc4'],layer_dims['out']),
    torch.nn.ReLU())

optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

mse = torch.nn.MSELoss()

for epoch in range(epochs):
    batch_ind = 0
    pred_val = model(X_val)
    val_loss = rmsle(pred_val, Y_val).data[0]
    pred_train = model(X_train)
    train_loss = rmsle(pred_train, Y_train).data[0]
    print("Epoch",epoch,": val loss", val_loss,", train loss:",train_loss)

    for step in range(steps_in_epoch):
        # Forward pass: compute predicted y by passing x to the model.
        X_train_batch = Variable(X_train.data[batch_ind:batch_ind + batch_size])
        Y_train_batch = Variable(Y_train.data[batch_ind:batch_ind + batch_size])
        y_pred = model(X_train_batch)

        # Compute and print loss.
        loss = mse(y_pred, Y_train_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        batch_ind += batch_size