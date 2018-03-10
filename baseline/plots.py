import numpy as np
import matplotlib.pyplot as plt

trng = np.array([0.9176, 0.8877, 0.9261, 0.9268, 0.9011, 0.9371, 0.9470, 
                 0.9440, 0.9430, 0.9514])

traw = np.array([0.9005, 0.8739, 0.8999, 0.9003, 0.8834, 0.9086, 0.9148, 
                 0.9119, 0.9081, 0.9201])

tavg = np.array([0.9005, 0.8982, 0.9140, 0.9271, 0.9198, 0.9340, 0.9321, 
                 0.9319, 0.9325, 0.9332])

xvals = [i+1 for i in xrange(len(trng))]
plt.figure()
plt.plot(xvals, trng, 'r-', label='train_acc')
plt.plot(xvals, traw, 'g-', label='test_acc')
plt.plot(xvals, tavg, 'b-', label='avg_test_acc')

plt.legend(loc=2)
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy")
plt.title("Perceptron Accuracy over iterations")
plt.savefig('./code/baseline/accuracy1.jpg')
plt.show()
