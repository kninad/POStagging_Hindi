import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

enVec = np.load('./data/en_vecs_csv.npy') 
hiVec = np.load('./data/hi_vecs_csv.npy') 

# Flip them around so they adhere to standard expected format of (N,D)
enVec = enVec.T
hiVec = hiVec.T

Ntest = 1000
X_train = hiVec[: -Ntest, :] # hi-vectors
Y_train = enVec[: -Ntest, :] # en-vectors
X_test = hiVec[-Ntest :, : ]
Y_test = enVec[-Ntest :, : ]

(N, D) = X_train.shape
(Ntst, D) = X_test.shape

from keras import backend as K
def l2loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=1)

num_epochs = 20
learn_rate = 1e-3
myBatchSize = 1

model = Sequential()
model.add(Dense(D, input_shape=(D,), use_bias=False, activation='linear'))

model.compile(loss=l2loss,
              optimizer=SGD(lr=learn_rate, momentum=0.9, nesterov=False),
              metrics=['accuracy'])
              
model.fit(X_train, Y_train, epochs=num_epochs, batch_size=myBatchSize)

print('\n Test set evaluation: \n')
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=1)
print(loss_and_metrics)

model.save('./code/myModel1.h5')



