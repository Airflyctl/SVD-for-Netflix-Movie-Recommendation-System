import numpy as np
import random
import matplotlib.pyplot as plt
# load the saved date set
training = np.loadtxt('C:/Users/airfl/Desktop/ML/homework2/hw2-training.txt')
testing = np.loadtxt('C:/Users/airfl/Desktop/ML/homework2/hw2-testing.txt')
matrix = np.load("C:/Users/airfl/Desktop/ML/homework2/matrix.npy")
oberved = np.load("C:/Users/airfl/Desktop/ML/homework2/oberved.npy")

K = 1000000
beta = 0.038
avgtest = 0
avgtrain = 0

# run several times of SGD
for times in range(1,11):
    k = 1
    H = np.load("C:/Users/airfl/Desktop/ML/homework2/H.npy")
    W = np.load("C:/Users/airfl/Desktop/ML/homework2/W.npy")
    
    while k <= K:
        i, j = random.choice(oberved)
        tk = np.inner(H[i], W[j]) - matrix[i][j]
        alpha = 0.97 / (k) ** (0.314)
        H[i], W[j] = H[i] - alpha * (tk * W[j] + beta * H[i]), W[j] - alpha * (tk * H[i] + beta * W[j])
        k += 1
    predictM = np.dot(H, np.transpose(W))

    # calculate the RSME of training data
    testerror = 0
    trainingerror = 0
    for i in range(len(testing)):
        userid, movieid, score, _ = testing[i]
        testerror += (score - predictM[int(userid) - 1][int(movieid) - 1]) ** 2
    testerror = (testerror / len(testing)) ** (0.5)
    avgtest += testerror

    # calculate the RSME of training data
    for i in range(len(training)):
        userid, movieid, score, _ = training[i]
        trainingerror += (score - predictM[int(userid) - 1][int(movieid) - 1]) ** 2
    trainingerror = (trainingerror / len(training)) ** (0.5)
    avgtrain += trainingerror
    print 'This is '+str(times)+ ' running,','The RSME of testing data is ' + str(testerror),  ', The RSME of Training data is ' + str(trainingerror)
    print '******************************************'

print 'The average RSME of testing data is '+ str(avgtest/10) ,  ', The average RSME of Training data is ' + str(avgtrain/10)
