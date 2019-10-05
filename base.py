import numpy as np
import matplotlib.pyplot as plt

class AIController:
    """ 
    2x3x2 neural 
    """
    def __init__(self):
        self.network = {}
        self.network[0] = {
            "W" : np.array(
                    [[0.1, 0.2, 0.3], 
                    [0.2, 0.1, 0.2]]),
            "b" : np.array(
                    [0.1, 0.2, 0.3])
        }
        self.network[1] = {
            "W" : np.array(
                    [[0.2, 0.3], 
                    [0.3, -0.2],
                    [-0.1, 0.1]]),
            "b" : np.array(
                    [0.3, 0.1])
        }

    def forword(self, inputMatrix):
        x1 = np.dot(inputMatrix, self.network[0]["W"]) + self.network[0]["b"]
        x2 = np.dot(x1, self.network[1]["W"]) + self.network[1]["b"]
        return x2
    
    def softmax(self, inputMatrix):
        maxInput = np.max(inputMatrix)
        expMatrix = np.exp(inputMatrix - maxInput);
        expSum = np.sum(expMatrix)
        return expMatrix/expSum

a = AIController()
print(a.forword(np.array([0.2, 0.1])))
print(a.softmax(np.sum(np.array([0.2, 0.1]))))
