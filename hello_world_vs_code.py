import pandas as pd
import numpy as np
nnp = np.array([[1,2,3,4], [11,12,13,14]])
kk = np.full_like(nnp, 9)
print(kk)
print(np.zeros((2,3,3)))
print(np.ones((2,4)))
print(np.full((2,2), 55))
print(np.random.rand(2,2))
print(np.random.random_sample(kk.shape))
print(np.random.randint(low= 245, high = 990, size = (2,2)))
print(np.random.randn(3,3))
print(np.array([1, 2, 3, 4])**4)
numbers = [1, 2, 3]
numbers_copy = numbers
numbers_right_copy = numbers.copy()
numbers.append(45)
print(numbers)
print(numbers_copy)
print(numbers_right_copy)

mylist = ["a", "b", "a", "c", "c"]
print(mylist)
new_list = list((dict.fromkeys(mylist)))
print(new_list)