import numpy as np
import random
import matplotlib.pyplot as plt
# create original maxtrix  AND  oberved set.
training = np.loadtxt('C:/Users/airfl/Desktop/ML/homework2/hw2-training.txt')
testing = np.loadtxt('C:/Users/airfl/Desktop/ML/homework2/hw2-testing.txt')
user, movie = 943 , 1682
matrix = np.zeros((user,movie))
oberved = []
for i in range(len(training)):
    userid, movieid, score, _ = training[i]
    matrix[int(userid)-1][int(movieid)-1] = score
    oberved.append([int(userid)-1,int(movieid)-1])
np.save("C:/Users/airfl/Desktop/ML/homework2/matrix.npy",matrix)
np.save("C:/Users/airfl/Desktop/ML/homework2/oberved.npy",oberved)

#SGD-----------------------------------------------------------------------------------------------------------------------
k = 1
K = 1000000
beta = 0.038
# initialize H and W
rank = int(user*0.1)
H = np.zeros((user,rank))
W = np.zeros((movie,rank))
for i in range(rank):
    H[i][i] = 1
    W[i][i] = 1

while k < K:
    i,j = random.choice(oberved)
    tk = np.inner(H[i],W[j]) - matrix[i][j]
    alpha = 0.97/(k)**(0.314)
    H[i],W[j] = H[i] - alpha*(tk*W[j] + beta*H[i]), W[j] - alpha*(tk*H[i] + beta*W[j])
    k += 1

# calculate the predict Matrix
predictM = np.dot(H, np.transpose(W))

#------------------------------------------------------------------------------------------------------------------
predicttest = [] # record the predict of testing date
testerror = 0
trainingerror = 0
# calculate the RSME of testing data
for i in range(len(testing)):
    userid, movieid, score, _  = testing[i]
    predicttest.append([int(userid), int(movieid),predictM[int(userid)-1][int(movieid)-1]])
    testerror += (score - predictM[int(userid)-1][int(movieid)-1])**2
testerror = (testerror /len(testing))**(0.5)

#-------------------------------------------------------------------------------------------------------------------------
# save predict testing data
np.savetxt("C:/Users/airfl/Desktop/ML/homework2/predicttest.txt",predicttest,"%3i")

















