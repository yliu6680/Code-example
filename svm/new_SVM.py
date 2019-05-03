# -*- coding: utf-8 -*-
from scipy.io import loadmat
from cvxopt import matrix, solvers
from itertools import compress
import numpy as np
from sklearn.metrics import confusion_matrix 

def loadMnist(file_path = "./"):
    image_data1 = loadmat(file_path + "new_MNIST_data.mat")
    return image_data1['train_samples'], image_data1['train_samples_labels'], image_data1['test_samples'], image_data1['test_samples_labels']

class SVM_classifier: 
    # kernal_options = [name, para]; opt_options = [C, learning_rate]
    def __init__(self, kernal_options=["polynomial", 6], opt_options=[10]): 
        self.kernal_options = kernal_options
        self.opt_options = opt_options
        self.p = kernal_options[1]
        self.sigma = kernal_options[1]
        self.C = opt_options[0]
        
    def linear(self, xi, xj):
        return np.dot(xi, xj)
    
    def polynomial(self, xi, xj):
        return (1 + np.dot(xi, xj)) ** self.p
    
    
    def gaussian(self, xi, xj):
        return np.exp(-np.linalg.norm(xi-xj)**2 / (2 * (self.sigma ** 2)))

    def compute_kernel(self, X, kernal_options):
        n_samples = X.shape[0]
        if kernal_options[0] == "linear":
            kernalFunc = self.linear
        elif kernal_options[0] == "polynomial":
            kernalFunc = self.polynomial
        elif kernal_options[0] == "gaussian":
            kernalFunc = self.gaussian
        else:
            raise ValueError("invalid kernal_options;")
        kernel_inner = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_inner[i, j] = kernalFunc(X[i], X[j])
        return kernel_inner      
    # train the model    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        print("compute kernel;")
        kernel_inner = self.compute_kernel(X, self.kernal_options) 
        print("finished;")

        # the matrcies are corresponded to the H,f,A,b,Aeq,Beq in the matlab qaudprog() function
        # these two matrices are for the maxize function in the equation (17) in Andrew ng's matrial. We use the -1 to transfer it to a minimize problem.
        P = matrix(np.outer(y,y) * kernel_inner)
        q = matrix(-1.0 * np.ones(n_samples))
        
        # matricies for the inequality 
        mat1 = np.identity(n_samples) * -1.0
        mat2 = np.identity(n_samples)
        G = matrix(np.vstack((mat1, mat2)))
        mat1 = np.zeros(n_samples)
        mat2 = np.ones(n_samples) * self.C
        h = matrix(np.hstack((mat1, mat2)))
        
        # matrices fo the equation
        A = matrix(y, (1,n_samples))
        b = matrix(0.0)
        
        # qp optimization
        print("Begin qp;")
        qp_result = solvers.qp(P, q, G, h, A, b)
        # langrange multiplier
        print("Finished;")
        alphas = np.ravel(qp_result['x']) 
        
        # get the support vectors
        sv_id = [] 
        for id, alpha in enumerate(alphas):
            if alpha > 1e-5 and alpha < self.C:
                sv_id.append(id)
        
        sv_alpha = alphas[sv_id]
        sv_X = X[sv_id]
        sv_y = y[sv_id]
        
        # compute the w values, using the equation 7.8
        print("computing w")
        w = np.zeros(n_features)
        for i in range(len(sv_id)):
            w += sv_alpha[i] * sv_y[i] * sv_X[i]
        
        # compute the b values, using the equation 7.18
        print("computing b")
        b = 0
        for i in range(len(sv_id)):
            b += sv_y[i]
            for j in range(len(sv_id)):
                b -= sv_alpha[j] * sv_y[j] * kernel_inner[i, j]
        b = b/len(sv_alpha)
        
        # store the values in classifier
        self.alphas, self.sv_id, self.kernel_inner = alphas, sv_id, kernel_inner
        self.w, self.b = w, b
        
    # predict the test data
    def predict(self, X):
        n_samples, n_features = X.shape
        
        ret_values = []
        ret_labels = []
        
        for i in range(n_samples):
            predict_values = 0
            predict_values += np.dot(self.w.T, X[i,:]) + self.b
            ret_values.append(predict_values)
            ret_labels.append(np.sign(predict_values))
        
        self.ret_values, self.ret_labels = ret_values, ret_labels
        
    def validate(self, X, y):
        self.predict(X)
        return confusion_matrix(y, self.ret_labels)
        