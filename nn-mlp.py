
# coding: utf-8

# In[167]:


import numpy as np

class Softmax:
    
    @staticmethod
    def activation(z):
        exps = np.exp(z - np.max(z))
        return exps / np.sum(exps)
 
    #@staticmethod
    #def prime(z):
        '''Assumes x is an instance of the Matrix class
           Derivative of the softmax function'''
        #softmax = Softmax.activation(z) 
        #return np.multiply(softmax, (1 - softmax))
        
class Tanh:        
    def activation(z):
        """ Compute the tanh function or its derivative.
        """
        return np.tanh(z)
    
    def prime(z):
        return 1 - np.square(Tanh.activation(z))
    

class Relu:
    @staticmethod
    def activation(z):
        z[z < 0] = 0
        return z

    @staticmethod
    def prime(z):
        z[z < 0] = 0
        z[z > 0] = 1
        return z


class Sigmoid:
    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def prime(z):
        return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))


class MSE:
    def __init__(self, activation_fn=None):
        """
        :param activation_fn: Class object of the activation function.
        """
        if activation_fn:
            self.activation_fn = activation_fn
        else:
            self.activation_fn = NoActivation

    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        return np.mean((y_pred - y_true)**2)

    @staticmethod
    def prime(y_true, y_pred):
        return y_pred - y_true

    def delta(self, y_true, y_pred):
        """
        Back propagation error delta
        :return: (array)
        """
        return self.prime(y_true, y_pred) * self.activation_fn.prime(y_pred)
 

class CrossEntropy:
    def __init__(self, activation_fn=Softmax):
        """
        :param activation_fn: Class object of the activation function.
        """
        if activation_fn:
            self.activation_fn = activation_fn
            print(self.activation_fn)
        else:
            self.activation_fn = NoActivation
            
    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_onehot, x):
        indices = np.argmax(y_onehot, axis = 1).astype(int)
        #x = Softmax.activation(x)
        predicted_probability = x[np.arange(len(x)), indices]
        log_preds = np.log(predicted_probability)
        loss = -1.0 * np.sum(log_preds) / len(log_preds)
        return loss
    
    def delta(self, y, a):
        """
        Cp_a, dC/da: the derivative of C w.r.t a
        ''a'' is the output of neurons
        ''y'' is the expected output of neurons
        """
        return (a - y) # delta
        
class NoActivation:
    """
    This is a plugin function for no activation.
    f(x) = x * 1
    """
    @staticmethod
    def activation(z):
        """
        :param z: (array) w(x) + b
        :return: z (array)
        """
        return z

    @staticmethod
    def prime(z):
        """
        The prime of z * 1 = 1
        :param z: (array)
        :return: z': (array)
        """
        return np.ones_like(z)


class Network:
    def __init__(self, dimensions, activations, dropout=False, learning_rate_decay=0):
        """
        :param dimensions: (tpl/ list) Dimensions of the neural net. (input, hidden layer, output)
        :param activations: (tpl/ list) Activations functions.
        Example of one hidden layer with
        - 2 inputs
        - 3 hidden nodes
        - 3 outputs
        layers -->    [1,        2,          3]
        ----------------------------------------
        dimensions =  (2,     3,          3)
        activations = (      Relu,      Sigmoid)
        """
        self.n_layers = len(dimensions)
        self.loss = None
        self.learning_rate = None
        self.dropout = dropout
        self.learning_rate_decay = learning_rate_decay
  
        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}

        # Activations are also initiated by index. For the example we will have activations[2] and activations[3]
        self.activations = {}
        for i in range(len(dimensions) - 1):
            self.w[i + 1] = np.random.randn(dimensions[i], dimensions[i + 1]) / np.sqrt(dimensions[i])
            self.b[i + 1] = np.zeros(dimensions[i + 1])
            self.activations[i + 2] = activations[i]
            
    def compute_dropout(self, activations, dropout_prob = 0.5):
        """Sets half of the activations to zero
        Params: activations - numpy array
        Return: activations, which half set to zero
        """
        # handle error
        if dropout_prob < 0 or dropout_prob > 1:
            dropout_prob = 0.5
            
        activations/=dropout_prob    
        mult = np.random.binomial(1, 0.5, size = activations.shape)
        activations*=mult
        return activations
    
    def _feed_forward(self, x, do_dropout = True):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """

        # w(x) + b
        z = {}

        # activations: f(z)
        a = {1: x}  # First layer has no activations as input. The input x is the input.

        for i in range(1, self.n_layers):
            # current layer = i
            # activation layer = i + 1
            z[i + 1] = np.dot(a[i], self.w[i]) + self.b[i]
            #if self.dropout and do_dropout:
                #if i < self.n_layers: 
            if i < (self.n_layers - 1):
                a[i + 1] = self.compute_dropout(self.activations[i + 1].activation(z[i + 1]))
            if i == (self.n_layers - 1):
                #a[i + 1] = self.activations[i + 1].activation(z[i + 1])
         
                a[i + 1] = z[i + 1]

        return z, a

    def _back_prop(self, z, a, y_true):
        """
        The input dicts keys represent the layers of the net.
        a = { 1: x,
              2: f(w1(x) + b1)
              3: f(w2(a2) + b2)
              }
        :param z: (dict) w(x) + b
        :param a: (dict) f(z)
        :param y_true: (array) One hot encoded truth vector.
        :return:
        """

        # Determine partial derivative and delta for the output layer.
        # delta output layer
        delta = self.loss.delta(y_true, a[self.n_layers])
        dw = np.dot(a[self.n_layers - 1].T, delta)

        update_params = {
            self.n_layers - 1: (dw, delta)
        }

        # In case of three layer net (two hidden layers) will iterate over i = 3 and i = 2
        # Determine partial derivative and delta for the rest of the layers.
        # Each iteration requires the delta from the previous layer, propagating backwards.
        for i in reversed(range(2, self.n_layers)):
            delta = np.dot(delta, self.w[i].T) * self.activations[i].prime(z[i])
            dw = np.dot(a[i - 1].T, delta)
            update_params[i - 1] = (dw, delta)

        for k, v in update_params.items():
            self._update_w_b(k, v[0], v[1])

    def _update_w_b(self, index, dw, delta):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        self.w[index] -= self.learning_rate * dw
        self.b[index] -= self.learning_rate * np.mean(delta, 0)

    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate, learning_rate_decay):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :param loss: Loss class (MSE, CrossEntropy etc.)
        :param epochs: (int) Number of epochs/iterations.
        :param batch_size: (int) Number of samples in minibatch
        :param learning_rate: (flt)
        """
        if not x.shape[0] == y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")
        # Initiate the loss object with the final activation function
        self.loss = loss(self.activations[self.n_layers])
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

        for i in range(epochs):
            # Shuffle the data
            seed = np.arange(x.shape[0])
            np.random.shuffle(seed)
            x_ = x[seed]
            y_ = y_true[seed]
            # minibatch
            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                z, a = self._feed_forward(x_[k:l])
                self._back_prop(z, a, y_[k:l])

            if (i + 1) % 10 == 0:
                _, a = self._feed_forward(x)
                #print("Loss:", self.loss.loss(y_true, a[self.n_layers]))
            # Learning rate decay    
            self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate * self.learning_rate_decay)))


    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        _, a = self._feed_forward(x)
        return a[self.n_layers]

if __name__ == "__main__":
    from sklearn import datasets
    import sklearn.metrics
    np.random.seed(1)
    #data = datasets.load_digits()
    import h5py
    with h5py.File('/Users/dyin/Desktop/Assignment-1-Dataset/train_128.h5','r') as H: data = np.copy(H['data'])
    with h5py.File('/Users/dyin/Desktop/Assignment-1-Dataset/train_label.h5','r') as H: label = np.copy(H['label'])

    from scipy import stats
    #x = stats.zscore(data)
    #data = (data - data.min()) / (data.max() - data.min())
    x = data[:1000]
    y = label[:1000]

    #x = data["data"]
    #y = data["target"]
    y = np.eye(10)[y]

    
    nn = Network((128, 96, 54, 10), (Relu, Tanh, Softmax))
    nn.fit(x, y, loss=CrossEntropy, epochs=2000, batch_size=50, learning_rate=1e-3, learning_rate_decay=0.0001)

    prediction = nn.predict(x)

    y_true = []
    y_pred = []
    for i in range(len(y)):
        y_pred.append(np.argmax(prediction[i]))
        y_true.append(np.argmax(y[i]))
    
    print(sklearn.metrics.classification_report(y_true, y_pred))
    
    
    #Run prediction over all records

    start = 5000  #pic first record from training data for validation
    end = 60000 ##pic last record from training data for validation
    x_test = data[start:end]
    y_test = label[start:end]

    t_results = nn.predict(x_test)

    correct = 0
    for i in range(len(y_test)):
        pred = np.argmax(t_results[i])
        true = y_test[i]
        if pred == true :
            correct += 1
    print('Accuracy = '+ str(correct/len(y_test)*100) + ' of ' + str(len(y_test)) + ' Samples')
    
    
    
    
    
from sklearn.model_selection import KFold # import KFold

kf = KFold(n_splits=10) # Define the split - into 2 folds 
kf.get_n_splits(x) # returns the number of splitting iterations in the cross-validator
print(kf) 
#KFold(n_splits=10, random_state=None, shuffle=False)

for train_indices, test_indices in kf.split(x):
    nn.fit(x[train_indices], y[train_indices], loss=CrossEntropy, epochs=2000, batch_size=50, learning_rate=1e-3, learning_rate_decay=0.0001)
    print(nn.score(x[test_indices], y[test_indices]))


# In[ ]:


from sklearn.model_selection import KFold # import KFold

kf = KFold(n_splits=10) # Define the split - into 2 folds 
kf.get_n_splits(x) # returns the number of splitting iterations in the cross-validator
print(kf) 
KFold(n_splits=10, random_state=None, shuffle=False)

for train_index, test_index in kf.split(X):
    print(“TRAIN:”, train_index, “TEST:”, test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]



# In[162]:




