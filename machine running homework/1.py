# import numpy as np
# d = np.loadtxt('./data.txt')

# print(d.mean(axis=1))

import numpy as np
array1 = np.array([3,4,5])
array2 = np.array([33,44,55])
array3 = np.array([33,24,15])
array4 = np.array([3,4,5])
array5 = np.array([34,4,5])

np.savez('./test',array1,array2,array3,array4,array5)
data=np.load('test.npz')
for i in range(len(data.files)):
    print(data['arr_%i' %i])