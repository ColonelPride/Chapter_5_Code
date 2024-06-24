import keras
import tensorflow as tf
import numpy as np

def predict_single(model,x): #get y prediction from single array of X
    #make 1D x into 3D (None,None,1)
    x=x.reshape(1,1,-1)
    y = model(x)
    #get 3D tensor back to 1D y array
    y=np.array(y[0][0][0])
    return y #1D array out

def predict_batch(model, X):  # get y prediction from single array of X
    # make 2D x into 3D (None,None,1)
    X_dim_1 = 1
    X_dim_2 = X.shape[0]
    X_dim_3 = X.shape[1]

    X = X.reshape(X_dim_1,X_dim_2,X_dim_3)
    y = model.predict(X,batch_size=None,verbose=None)
    # get 3D tensor back to 1D y array
    y = y[0].reshape(-1)
    return y #output 1D  array