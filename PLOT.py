import numpy as np
import random
import matplotlib.pyplot as plt

K = 1000000

testerrorl = np.load("C:/Users/airfl/Desktop/ML/homework2/testerrorl.npy")
trainingerrorl = np.load("C:/Users/airfl/Desktop/ML/homework2/trainingerrorl.npy")

t = np.arange(20)
plt.title('Testing and Training Error',size = '30')
plt.plot(t,testerrorl,color="blue", linewidth=2.5,label="testingerror")
plt.plot(trainingerrorl,color="red", linewidth=2.5,label="trainingerror")
plt.legend(loc='upper right')
plt.yticks(np.arange(0.8,1.5,0.05))
plt.grid(True)
plt.show()