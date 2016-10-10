import numpy as np
import random
import matplotlib.pyplot as plt

# load the saved data
training = np.loadtxt('C:/Users/airfl/Desktop/ML/homework2/hw2-training.txt')
testing = np.loadtxt('C:/Users/airfl/Desktop/ML/homework2/hw2-testing.txt')
matrix = np.load("C:/Users/airfl/Desktop/ML/homework2/matrix.npy")
oberved = np.load("C:/Users/airfl/Desktop/ML/homework2/oberved.npy")

#SGD-----------------------------------------------------------------------------------------------------------------------
# set the parameter
k = 1
K = 1000000
beta = 0.038
user, movie = 943 , 1682
# initialize H and W

rank = int(user*0.01)
H = np.zeros((user,rank))
W = np.zeros((movie,rank))
for i in range(rank):
    H[i][i] = 1
    W[i][i] = 1

#--------------------------------------------------------------------------
# list of testerror and trainingerror , using these two to plot in RMS.py
testerrorl = []
trainingerrorl = []

while k <= K:
    i,j = random.choice(oberved)
    tk = np.inner(H[i],W[j]) - matrix[i][j]
    alpha = 0.97/(k)**(0.314)
    H[i],W[j] = H[i] - alpha*(tk*W[j] + beta*H[i]), W[j] - alpha*(tk*H[i] + beta*W[j])

    if k % 50000 == 0: # Every 50000 steps to choose one testing error and traning error
        predictM = np.dot(H, np.transpose(W))
        testerror, trainingerror = 0, 0
        for i in range(len(testing)):
            userid, movieid, score, _ = testing[i]
            testerror += (score - predictM[int(userid) - 1][int(movieid) - 1]) ** 2
        testerrorl.append( (testerror / len(testing)) ** (0.5))

        for i in range(len(training)):
            userid, movieid, score, _ = training[i]
            trainingerror += (score - predictM[int(userid) - 1][int(movieid) - 1]) ** 2
        trainingerrorl.append ( (trainingerror / len(training)) ** (0.5))

    k += 1

# save the list of testerror and trainingerror in order to load in RMS.py
np.save("C:/Users/airfl/Desktop/ML/homework2/testerrorl.npy",testerrorl)
np.save("C:/Users/airfl/Desktop/ML/homework2/trainingerrorl.npy",trainingerrorl)

#-------------------------------------------------------------------------------------------------------------------------



















