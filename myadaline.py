import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import seed
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers=('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    x3_min, x3_max = X[:, 2].min()-1, X[:, 1].max()+1
    xx1, xx2, xx3 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                np.arange(x2_min, x2_max, resolution),
                np.arange(x3_min, x3_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel(), xx3.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    # plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    # plt.xlim(xx1.min(), xx1.max())
    # plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    fig =plt.figure()
    ax=fig.add_subplot(projection='3d')
    for idx, cl in enumerate(np.unique(y)):

        ax.scatter(xs=X[y==cl, 0], ys=X[y==cl, 1], zs=X[y==cl, 2],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
def RGBtocsv():
    
    ubaux = "C:/Users/Usuario/Documents/SEA6/programacion/python/machine_learning"
    # ubaux="I:\documents_Virtual\PROGRAMACION Y PROTOTIPOS\python\machine_learning"
    arr=np.loadtxt(ubaux + "/{}RGB".format("recorte"))
    valores=['M','N','R','H','N','H','R','M','R','M','M','H','H','H','H','R']
    dict={'H':0, 'R':1, 'M':2, 'N':3}
    arr1 = np.empty([16384,4])
    i=0
    j=0
    for item in arr:
        if i!=0 and i % 1024 == 0 :
            j+=1
        arr1[i][0]=item[0]
        arr1[i][1]=item[1]
        arr1[i][2]=item[2]
        arr1[i][3]=dict[valores[j]]
        i+=1
    return arr1
class AdalineGD (object):
    """ADAptive LInear NEuron classifier.
    
    Parameters
    __________
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
        
    Attributes
    __________
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.
    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit (self, X, y):
        """ Fit training data.
        
        Parameters
        __________
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors,
            where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
            
        Returns
        _______
        self: object
        """
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            # la actualización de los pesos se hace cada iteración (epoch)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    def net_input(self, X):
        """Calculate the net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X)>=0.0, 1, -1)

class AdalineSGD(object):
    """ADAptive LInear NEuron classifier
    Parameters
    __________
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Passes over the training dataset.
    Attributes
    ----------
    w_: 1d-array
        Weights after fitting.
    errors_ :list
        Number of misclassifications in every epoch.
    shuffle: bool (default:True)
        Shuffles training data every epoch
        if True to prevent cycles.
    random_state: int (default: None)
        Set random state for shuffling
        and initializing the weights.
    """
    def __init__(self, eta = 0.01, n_iter=10,
                shuffle = True, random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)
    def fit(self, X, y):
        """Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples
            is the number of samples and
            n_features is the number of features.
        y : array-like, shape=[n_samples]
            Target values.
            
        Returns
        -------
        self: object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                # la actualización de los pesos se hace con cada muestra
                cost.append(self._update_weights(xi,target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self
    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self
    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]
    def _initialize_weights(self, m):
        """Initialize weights to zeros"""
        self.w_=np.zeros(1+m)
        self.w_initialized = True
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(xi)
        error=(target-output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)
    def predict(self, X):
        """Returno class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)


if __name__ =='__main__':
    arr = RGBtocsv()
    df = pd.DataFrame(arr)
    # df = pd.read_csv('https://archive.ics.uci.edu/ml/'
    #     'machine-learning-databases/iris/iris.data', header = None)
    y=df.iloc[0:4096, 3].values
    # y=np.where(y=='Iris-setosa', -1, 1)
    X=df.iloc[0:4096, [0,1,2]].values    
    """ en el siguiente código de ve la relación con el "learning rate" """
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    # ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    # ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    # ax[0].set_xlabel('Epochs')
    # ax[0].set_ylabel('log(Sum-squared-error)')
    # ax[0].set_title('Adaline-Learning rate 0.01')
    # ada2 = AdalineGD(n_iter=10, eta = 0.0001).fit(X, y)
    # ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    # ax[1].set_xlabel('Epochs')
    # ax[1].set_ylabel('Sum-squared-error')
    # ax[1].set_title('Adaline-Learning rate 0.0001')    
    # plt.show()
    """A continuación se utiliza la estandarización para mejora la velocidad de convergencia"""
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0]-X[:,0].mean())/X[:,0].std()
    X_std[:,1] = (X[:,1]-X[:,1].mean())/X[:,1].std()
    X_std[:,2] = (X[:,2]-X[:,2].mean())/X[:,2].std()
    """ADALINEGD"""
    # ada = AdalineGD(n_iter=15, eta=0.01)
    # ada.fit(X_std, y)
    # plot_decision_regions(X_std, y, classifier=ada)
    # plt.title('Adaline-Gradient Descent')
    # plt.xlabel('sepal length [standarized]')
    # plt.ylabel('petal length [standarized]')
    # plt.legend(loc='upper left')
    # plt.show()
    # plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    # plt.xlabel('Epochs')
    # plt.ylabel('Sum-squared-error')
    # plt.show()
    """ADALINESGD  Stochastic (ajusta con cada muestra)"""
    
    ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline-Stochastic Gradient Descent')
    plt.xlabel('sepal length [standarized]')
    plt.ylabel('petal length [standarized]')
    plt.legend(loc='upper left')
    plt.show()
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.show()