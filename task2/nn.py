import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_args():
    p = argparse.ArgumentParser(
        description="python nn.py NInput NHidden NOutput train.csv.gz test.csv.gz"
    )
    p.add_argument("NInput", type=int)
    p.add_argument("NHidden", type=int)
    p.add_argument("NOutput", type=int)
    p.add_argument("train_path")
    p.add_argument("test_path")
    return p.parse_args()

# Load the text
def loadFile(fName):
    R = np.loadtxt(fName, delimiter=',', dtype=np.int32, ndmin=2, skiprows=1)
    y_raw = R[:, 0] # collects all labels, 1 on each row
    x_raw = R[:, 1:785] # collects all 784 pixel values for each row
    return x_raw, y_raw

# Normalise the pixels to [0,1]
def normaliseX(x_raw):
    x = np.asarray(x_raw) # uses numpy array to decrease time cost as all happens inside C
    return x / 255.0 # returns the array with each element divided by 255.0 to normalise from 0 to 1

# changes [3,7,1,3,2] to [[0,0,0,1,0,0,0,0,0,0],[0,..],..]
def normaliseY(y_raw, num_classes=10):
    y = np.asarray(y_raw, dtype=np.int32) # convert y_raw to an array of int32
    I = np.eye(num_classes) # returns a 10x10 identity matrix
    return I[y] # picks row of matrix I for each value in y

def make_batches(N, batch_size, rng): # yields a list of indexes of batch size
    indicies = rng.permutation(N) # returns a random list of indicies with values [0,N-1]
    for start in range(0,N,batch_size): # loops over a list where if batch size = 20, [0,20,40,60,..,N]
        yield indicies[start:start+batch_size] # returns the list of that batch, stores where it is up to in loop and waits for next call to return the next batch
    
def transpose(x_rows, y_rows, indexes):
    # collect batches of only the specified indexes
    x_batch = x_rows[indexes]
    y_batch = y_rows[indexes]
    # transpose them from x = (N, 784), y = (N, 10) to (784, N) and (10, N) to make it better for the math formulas
    x_trans = x_batch.T
    y_trans = y_batch.T
    return x_trans, y_trans

# initialises weights and bias for each neuron, input is number of neurons on each layer ouptpt is a dict with all weights and biases
def init_weights_and_bias(n_in, n_hidden, n_out, rng=None):
    if rng is None:
        rng = np.random.default_rng(69) 
    # Xavier/Glorot scale for sigmoid, good area to randomly initialse weights
    limit_W1 = np.sqrt(1/n_in) # weight scale for connection n_in --> n_hidden
    limit_W2 = np.sqrt(1/n_hidden) # weight scale for connection n_hidden --> n_out
    # Random uniform distribution of weights using limits above to every connection in neural network
    w1 = rng.uniform(-limit_W1, limit_W1, size=(n_hidden, n_in)).astype(np.float32)
    w2 = rng.uniform(-limit_W2, limit_W2, size=(n_out, n_hidden)).astype(np.float32)
    # Sets each hidden and output neuron bias to 0
    b1 = np.zeros((n_hidden, 1))
    b2 = np.zeros((n_out, 1))
    return {"w1":w1,"b1":b1,"w2":w2,"b2":b2}

def sigmoid(z):
    z = np.clip(z, -40.0, 40.0) # prevents exp overflow
    return 1 / (1+np.exp(-z)) # returns sigmoid values of Z array

def forward_prop(w_and_b, x_batch):
    # hidden pre-activation
    Z1 = w_and_b["w1"] @ x_batch + w_and_b["b1"] # @ is matrix multiplication * is elementwise
    # hidden activation
    A1 = sigmoid(Z1)
    # output pre-activation
    Z2 = w_and_b["w2"] @ A1 + w_and_b["b2"]
    # output activation
    A2 = sigmoid(Z2)
    # predictions per sample
    pred = np.argmax(A2, axis=0)
    return {"z1":Z1, "z2": Z2, "a1": A1, "a2": A2, "pred": pred}

def back_prop(x_batch, y_batch, z_and_a, w_and_b):
    # output layer error
    s2 = z_and_a["a2"] - y_batch # simplified as using sigmoid + cross entropy (shape is (10, m))
    # hidden layer error
    s1 = w_and_b["w2"].T @ s2 * z_and_a["a1"] * (1-z_and_a["a1"]) # z_and_a["a1"] * (1-z_and_a["a1"]) is derivative of sigmoid, shape is (n_hidden, m)
    # output layer gradients
    m = y_batch.shape[1]
    dw2 = (1 / m) * s2 @ z_and_a["a1"].T # shape (10, NHidden)
    db2 = (1 / m) * np.sum(s2, axis=1, keepdims=True) # shape (NOutput, 1)
    # hidden layer gradients
    dw1 = (1 / m) * s1 @ x_batch.T # shape (NHidden, NInput)
    db1 = (1 / m) * np.sum(s1, axis=1, keepdims=True) # shape (NHidden, 1)
    return {"dw1": dw1, "db1" : db1, "dw2" : dw2, "db2" : db2} 
 

def update_parameters(w_and_b, back_prop, learning_rate):
    w_and_b["w1"] -= learning_rate * back_prop["dw1"] # Updates W1 to reflect new weights
    w_and_b["w2"] -= learning_rate * back_prop["dw2"] # Updates B1 to reflect new Biases
    w_and_b["b1"] -= learning_rate * back_prop["db1"] # Updates W2 to reflect new weights
    w_and_b["b2"] -= learning_rate * back_prop["db2"] # Updates B2 to reflect new Biases
    return w_and_b

# Trains one epoch
def train_one_epoch(x_train, y_train, w_and_b, batch_size, learning_rate, epoch):
    total_loss = 0.0 # initialise loss to 0
    num_batches = 0 # initialise num batches to 0
    rng = np.random.default_rng(69+epoch) # rng is 69+epoch to have fresh seed each run but ensure reproducability
    for indicies in make_batches(len(x_train), batch_size, rng):
        Xb, Yb = transpose(x_train, y_train, indicies) # transpose x and y batch to match math formulas
        z_and_a = forward_prop(w_and_b, Xb) # runs forward propagation algorithm
        dw_and_db = back_prop(Xb, Yb, z_and_a, w_and_b) # runs back propagation algorithm
        w_and_b = update_parameters(w_and_b, dw_and_db, learning_rate) # updates the parameters
        num_batches += 1 # adds counter to num batches
    return w_and_b

def epoch_accuracy(x_test, y_test, w_and_b, batch_size): # cacluates accuracy of one epoch
    num_correct = 0 # initialise num correct to 0
    for indicies in make_batches(len(x_test), batch_size, rng=np.random.default_rng(69)):
        Xb, Yb = transpose(x_test, y_test, indicies) # transpose x and y batch to match math formulas
        pred = forward_prop(w_and_b, Xb)["pred"] # runs forward propagation algorithm to cacluate a prediction
        true = np.argmax(Yb, axis=0) # finds the true value in y
        num_correct += np.sum(pred == true) # checks if prediction == true and +1 to num correct if so
    return num_correct / y_test.shape[0] # returns accuracy from 0-1

def train(x_train, y_train, x_test, y_test, w_and_b, batch_size, learning_rate, epochs): # trains neural network using specified hyper parameters
    accuracy_array = [] # initialises accuracy array to empty
    for i in range(epochs): # loops over number of epochs
        w_and_b = train_one_epoch(x_train, y_train, w_and_b, batch_size, learning_rate, epoch=i) # trains one epoch
        acc = epoch_accuracy(x_test, y_test, w_and_b, batch_size) # calculates acuracy after one is trained
        accuracy_array.append(acc) # append the accuracy to the list
    max_test_acc = np.max(accuracy_array) # finds the max test accuracy
    return max_test_acc, accuracy_array # returns the accuracy array and the max acc

def plot_task_1(accuracy_array, epochs):
    ## Plot learning Curve
    # create x axis based on epochs:
    x = [i for i in range(epochs)]
    y = [i*100 for i in accuracy_array] # accuracy % 0-100
    plt.plot(x, y, marker='o') # line with circular markers
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Neural Network Learning Curve")
    plt.grid(True) # adds a grid
    plt.show() # display window   

def task1(xtr, ytr, xte, yte, params, batch_size, learning_rate, epochs): # runs and plots experiment 1
    max_test_acc, accuracy_array = train(xtr, ytr, xte, yte, params, batch_size, learning_rate, epochs)
    plot_task_1(accuracy_array, epochs)
    print(f"Task 1 Max test accuracy: {max_test_acc}")

def task2(xtr, ytr, xte, yte, NIn, NHid, NOut, batch_size, epochs): # runs and plots experiment 2
    learning_rates = [0.001, 0.01, 1.0, 10.0, 100.0] # initialise learning rates
    acc_graphs = {} # dictionary of {lr: accuracy array, ...}
    for lr in learning_rates: # runs training and testing over each learning rate
        params = init_weights_and_bias(NIn, NHid, NOut)
        max_test_acc, accuracy_array = train(xtr, ytr, xte, yte, params, batch_size, lr, epochs)
        acc_graphs[lr] = accuracy_array
        print(f"Max Accuracy for LR={lr}: {max_test_acc}") # print max accuracy for each lr
    # Plotting
    x = [i for i in range(epochs)]
    for lr, accs in acc_graphs.items(): # plots curves ontop of eachother
        y = [i*100 for i in accs]
        plt.plot(x, y, marker='o', label=f"lr={lr}")
        plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Neural Network Learning Curve")
    plt.grid(True) # adds a grid
    plt.legend()
    plt.show() # display window  
    
def task3(xtr, ytr, xte, yte, NIn, NHid, NOut, learning_rate, epochs): # runs and plots experiment 3
    batch_sizes = [1, 5, 20, 100, 300] # initialises batch sizes
    max_accs = []
    for bs in batch_sizes: # runs train and test for each batch size
        params = init_weights_and_bias(NIn, NHid, NOut)
        max_test_acc, accuracy_array = train(xtr, ytr, xte, yte, params, bs, learning_rate, epochs) 
        max_accs.append(max_test_acc)
        print(f"Max Accuracy for Batch Size={bs}: {max_test_acc}") # retrive max accuracy for each batch size
    y = [i*100 for i in max_accs] 
    plt.plot(batch_sizes, y, marker='o') # plots max acc vs batch size, does not use accuracy array
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy (%)")
    plt.title("Neural Network Learning Curve")
    plt.grid(True) # adds a grid
    plt.show() # display window  


def main():
    # parses the arguments
    args = parse_args()
    # Loads the train File
    x_train, y_train = loadFile(args.train_path)
    x_test, y_test = loadFile(args.test_path)
    # Normalise the pixel values
    xtr = normaliseX(x_train)
    xte = normaliseX(x_test)
    # Normalise the labels
    ytr = normaliseY(y_train)
    yte = normaliseY(y_test)
    # inititalise all weights and biases
    params = init_weights_and_bias(args.NInput, args.NHidden, args.NOutput)
    # Task 1
    task1(xtr, ytr, xte, yte, params, batch_size=20, learning_rate=3.0, epochs=30)
    # Task 2
    task2(xtr, ytr, xte, yte, args.NInput, args.NHidden, args.NOutput, batch_size=20, epochs=30)
    # Task 3
    task3(xtr, ytr, xte, yte, args.NInput, args.NHidden, args.NOutput, learning_rate=3.0, epochs=30)
    # Task 4
    params = init_weights_and_bias(args.NInput, args.NHidden, args.NOutput)
    task1(xtr, ytr, xte, yte, params, batch_size=100, learning_rate=1.0, epochs=100)



main()

# RUN: python nn.py 784 30 10 fashion-mnist_train.csv.gz fashion-mnist_test.csv.gz