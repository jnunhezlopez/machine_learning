from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, test_idx = None, resolution=0.2):
    markers=('s', 'x', 'o', '^', 'v')
    # colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    colors = ('red', 'black', 'yellow', 'gray', 'cyan')
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
        if test_idx:
            X_test, y_test = X[test_idx, :], y[test_idx]
            ax.scatter(X_test[:,0], X_test[:,1],X_test[:,2], facecolors='none',
                edgecolors='b', alpha=1.0, linewidths=1, marker='o', s=55, 
                label='test set')
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
def RGBtocsv():
    
    ubaux = "C:/Users/Usuario/Documents/SEA6/python/machine_learning"
    
    # ubaux="I:\documents_Virtual\PROGRAMACION Y PROTOTIPOS\python\machine_learning"
    arr=np.loadtxt(ubaux + "/{}RGB".format("recorte"))
    valores=['M','N','R','H','N','H','R','M','R','M','M','H','H','H','H','R']
    dict={'R':0, 'N':1, 'H':2, 'M':3}
    dimRGB=16
    casillas = 16
    arr1 = np.empty([casillas*dimRGB**2,4])
    i=0
    j=0
    for item in arr:
        if i!=0 and i % dimRGB**2 == 0 :
            j+=1
        arr1[i][0]=item[0]
        arr1[i][1]=item[1]
        arr1[i][2]=item[2]
        arr1[i][3]=dict[valores[j]]
        i+=1
    return arr1

if __name__ =='__main__':
    arr = RGBtocsv()
    df = pd.DataFrame(arr)
    y=df.iloc[0:1000, 3].values
    X=df.iloc[0:1000, [0,1,2]].values  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    ppn=Perceptron(max_iter=40, eta0=0.1, random_state=0)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    # print('Misclassified samples: %d' % (y_test != y_pred).sum())
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn,
                    test_idx=range(700,1000))
    # plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn,
    #                 test_idx=None)                    
    plt.legend(loc='upper left')
    plt.show()