import numpy as np

a = np.array([0.7425455420603501, 0.7171860233999476, 0.7394144770477795, 0.7143134482388142, 0.7363703529542696])
b = np.array([0.7425455420603501, 0.6171860233999476, 0.6394144770477795, 0.6143134482388142, 0.6363703529542696])
c = np.array([0.7425455420603501, 0.6171860233999476, 0.5394144770477795, 0.6143134482388142, 0.6363703529542696])


print(np.std(a))
print(np.std(b))
print(np.std(c))
print(c[(4-4):4])
d = c[(4-4):4]
print(d)
# print(np.std(d) < 0.02)
print(np.std(c[(4-4):4]) < 0.02)

