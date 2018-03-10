
import numpy as np
import pickle

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler
from keras import backend as K

ROOT_DIR = '/home/ninad/Desktop/NlpProj/'
CURR_DIR = ROOT_DIR + 'code/mikolov/GSlim/'

enVec = np.load(CURR_DIR + 'new_envecs_csv.npy') 
hiVec = np.load(CURR_DIR + 'new_hivecs_csv.npy') 
# Flip them around so they adhere to standard expected format of (N,D)
enVec = enVec.T
hiVec = hiVec.T


X_train = hiVec # hi-vectors
X_test = hiVec[-100 :, : ]

Y_train = enVec # en-vectors
Y_test = enVec[-100 :, : ]

(N, D) = X_train.shape
(Ntst, D) = X_test.shape

def l2loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=1)

num_epochs = 500
myBatchSize = 4000
learn_rate = 1e-3
decay_rate = 5e-7
my_sgd = SGD(lr=learn_rate, momentum=0.9, decay=decay_rate)
my_adam = Adam(lr=learn_rate, decay=decay_rate)

model = Sequential()
model.add(Dense(D, input_shape=(D,), use_bias=False, activation='linear'))

model.compile(loss=l2loss,
              #optimizer=my_sgd,
              optimizer=my_adam)
              
model.fit(X_train, Y_train, epochs=num_epochs, batch_size=myBatchSize)
model.save(CURR_DIR + 'myModel200.h5')
print("Model saved \n")

print('Test set evaluation: \n')
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=1)
print(loss_and_metrics)

x = model.layers[0].get_weights()
W = np.asarray(x)
W = W[0,:,:]
np.save(CURR_DIR+'W_sgd.npy', W)




########### ALTERNATE WAY ############
# Directly compute the matrix using inverse
# X*W = Y
 
#~ B = np.dot(X_train.T, Y_train)
#~ A = np.dot(X_train.T, X_train)
#~ A_inv = np.linalg.inv(A)

#~ W_linalg = np.dot(A_inv, B)

#~ np.save(CURR_DIR + 'Weight_matrix_linalg.npy', W_linalg)










