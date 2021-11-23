import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W

def sigmoid(z):
 return (1.0 / (1.0 + np.exp(-z)))


def preprocess():
    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    
    tmp = []
    for i in range(10):
        idx = 'train'+ str(i)
        train_mat = mat[idx]
        labels = np.full((train_mat.shape[0],1),i)
        labeled_train_mat = np.concatenate((train_mat,labels),axis=1)
        tmp.append(labeled_train_mat)

    all_labeled_train = np.concatenate((tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5],tmp[6],tmp[7],tmp[8],tmp[9]), axis=0)
    
    np.random.shuffle(all_labeled_train)
    
    labeled_train = all_labeled_train[0:50000,:]
    train_data    = labeled_train[:,0:784]
    train_label   = labeled_train[:,784]

    train_data = train_data / 255.0

    labeled_validation = all_labeled_train[50000:60000,:]
    validation_data    = labeled_validation[:,0:784] 
    validation_label   = labeled_validation[:,784]

    validation_data = validation_data / 255.0 


    tmp1 = []
    for i in range(10):
        idx = 'test'+ str(i)
        test_mat = mat[idx]
        labels = np.full((test_mat.shape[0],1),i)
        labeled_test_mat = np.concatenate((test_mat,labels),axis=1)
        tmp1.append(labeled_test_mat)

    all_labeled_test = np.concatenate((tmp1[0],tmp1[1],tmp1[2],tmp1[3],tmp1[4],tmp1[5],tmp1[6],tmp1[7],tmp1[8],tmp1[9]), axis=0)

    np.random.shuffle(all_labeled_test)
    
    test_data    = all_labeled_test[:,0:784]
    test_label   = all_labeled_test[:,784]

    test_data = test_data / 255.0


    combined  = np.concatenate((train_data, validation_data),axis=0)
    reference = combined[0,:]
    boolean_value_columns = np.all(combined == reference, axis = 0)
    featureCount=0

    for i in range(len(boolean_value_columns)):
        if boolean_value_columns[i]==False:
            featureCount += 1
    print("Total number of selected features : ", featureCount)
    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    
    obj_val = 0
    n = data.shape[0]

    biases1 = np.full((n,1), 1)
    data_bias = np.concatenate((biases1, data),axis=1)
    aj = np.dot(data_bias, np.transpose(w1))
    zj = sigmoid(aj)
    m = zj.shape[0]
    
    biases2 = np.full((m,1), 1)
    zj_bias = np.concatenate((biases2, zj), axis=1)
    bl = np.dot(zj_bias, np.transpose(w2))
    ol = sigmoid(bl)
    yl = np.full((n, n_class), 0)

    for i in range(n):
        trueLabel = training_label[i]
        yl[i][trueLabel] = 1
    
    yl_prime = (1.0-yl)
    ol_prime = (1.0-ol)
    lol = np.log(ol)
    lol_prime = np.log(ol_prime)
    error = np.sum( np.multiply(yl,lol) + np.multiply(yl_prime,lol_prime) )/((-1)*n)
    delta = ol- yl
    gradient_w2 = np.dot(delta.T, zj_bias)
    temp = np.dot(delta,w2) * ( zj_bias * (1.0-zj_bias)) 
    gradient_w1 = np.dot( np.transpose(temp), data_bias)
    gradient_w1 = gradient_w1[1:, :]
    regularization =  lambdaval * (np.sum(w1**2) + np.sum(w2**2)) / (2*n)
    obj_val = error + regularization
    gradient_w1_reg = (gradient_w1 + lambdaval * w1)/n
    gradient_w2_reg = (gradient_w2 + lambdaval * w2)/n
    obj_grad = np.concatenate((gradient_w1_reg.flatten(), gradient_w2_reg.flatten()), 0)
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    n = data.shape[0]

    biases1 = np.full((n,1),1)
    data = np.concatenate((biases1, data), axis=1)

    aj = np.dot(data, w1.T)
    zj = sigmoid(aj)
    
    m = zj.shape[0]
    
    biases2 = np.full((m,1), 1)
    zj = np.concatenate((biases2, zj), axis=1)

    bl = np.dot(zj, w2.T)
    ol = sigmoid(bl)

    labels = np.argmax(ol, axis=1)

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
