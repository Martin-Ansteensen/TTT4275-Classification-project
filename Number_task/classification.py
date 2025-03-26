import matplotlib.pyplot as plt
import numpy as np


def reshape_matrix(mat):
    reshapedMat = []
    
    for i in range(np.size(mat,0)):
        reshapedMat.append(mat[i,:].reshape(28,28))

    return np.array(reshapedMat)

# Extract data from csv files

trainv = np.loadtxt("/home/cyprian/vscopium/TTT4275-Classification-project/Number_task/numbers_data/trainv.csv", dtype=np.uint8, delimiter=',')
testv = np.loadtxt("/home/cyprian/vscopium/TTT4275-Classification-project/Number_task/numbers_data/testv.csv", dtype=np.uint8, delimiter=',')
trainlab = np.loadtxt("/home/cyprian/vscopium/TTT4275-Classification-project/Number_task/numbers_data/trainlab.csv", dtype=np.uint8, delimiter=',')
testlab = np.loadtxt("/home/cyprian/vscopium/TTT4275-Classification-project/Number_task/numbers_data/testlab.csv", dtype=np.uint8, delimiter=',')

reshapedTrainv = reshape_matrix(trainv)
reshapedTestv = reshape_matrix(testv)

neareast_neighbour = np.loadtxt('Number_task/out2.txt', dtype=np.uint32, delimiter=",")
print(neareast_neighbour)

predictions = []
for elm in neareast_neighbour:
    predictions.append(trainlab[elm])

predictions = np.array(predictions)
print(predictions)
print(testlab)

print(np.sum(predictions == testlab))