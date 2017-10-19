import numpy as np
from data_utils import DataUtils as du
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

#TOTAL_DATASET_SIZE = 10887
epochs = 10000
batch_size = 64
layer_dims = {"in": 13, "fc1": 10, "fc2": 10, "fc3": 10,"fc4": 10, "out": 1}
dtype = torch.Tensor

def rmsle(y_pred,y_true):
    log_err = torch.log(y_pred + 1) - torch.log(y_true + 1)
    squared_le = torch.pow(log_err,2)
    mean_sle = torch.mean(squared_le)
    root_msle = torch.sqrt(mean_sle)
    return (root_msle)

if __name__ == '__main__':
    _,_,_,X_train, Y_train, Y_train_log,X_val, Y_val,X_test,test_date_df = du.get_processed_df('data/train.csv', 'data/test.csv')

    train_data_size = X_train.shape[0]
    print(train_data_size)
    steps_in_epoch = train_data_size // batch_size

    #Dividing data to train and val datasets, conversion from DF to numpyarray
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    Y_train_log = np.array(Y_train_log)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    X_test = np.array(X_test)

    # Create random Tensors to hold inputs and outputs, and wrap them in Variables.
    X_train = Variable(torch.Tensor(X_train))
    Y_train = Variable(torch.Tensor(Y_train))
    Y_train_log = Variable(torch.Tensor(Y_train_log))
    X_train_batch = Variable(torch.randn(batch_size, 13))
    Y_train_batch = Variable(torch.randn(batch_size))
    X_val = Variable(torch.Tensor(X_val))
    Y_val = Variable(torch.Tensor(Y_val))
    X_test = Variable(torch.Tensor(X_test))

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

    optimizer = torch.optim.Adadelta(model.parameters(),lr=0.1)

    mse = torch.nn.MSELoss()

    best_val_error = 1000
    best_train_error = 1000
    train_err_his = np.zeros(epochs)
    val_err_his = np.zeros(epochs)

    for epoch in range(epochs):
        batch_ind = 0
        info = ""

        # Counting error on val set
        pred_val = model(X_val)
        val_loss = rmsle(pred_val, Y_val).data[0]
        val_err_his[epoch] = val_loss
        if val_loss < best_val_error:
            best_val_error = val_loss
            info = "Val error has improved!"
            torch.save(model,"saved_model.mdl")

        # Counting error on train set
        pred_train = model(X_train)
        train_loss = rmsle(pred_train, Y_train).data[0]
        train_err_his[epoch] = train_loss
        if train_loss < best_train_error:
            best_train_error = train_loss
            info += "Train error has improved!"

        print("Epoch",epoch,": val loss", val_loss,", train loss:",train_loss,info)

        for step in range(steps_in_epoch):
            # Forward pass: compute predicted y by passing x to the model.
            X_train_batch = Variable(X_train.data[batch_ind:batch_ind + batch_size])
            Y_train_batch = Variable(Y_train.data[batch_ind:batch_ind + batch_size])
            y_pred = model(X_train_batch)

            # Compute and print loss.
            loss = mse(y_pred, Y_train_batch)
            #print(torch.mean(loss).data)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            batch_ind += batch_size

    model = torch.load("saved_model.mdl")
    predictions = model(X_test)
    np.savetxt("predictions.csv", np.array(predictions.data.numpy()), delimiter=",")
    plt.plot(val_err_his)
    plt.plot(train_err_his)
    plt.show()