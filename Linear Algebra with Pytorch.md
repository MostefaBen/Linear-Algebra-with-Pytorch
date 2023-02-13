# Linear Algebra with Pytorch


```python
# import libraries
import torch
import numpy as np
```


```python
# matrix
A = torch.arange(6).reshape(3, 2)
A
```




    tensor([[0, 1],
            [2, 3],
            [4, 5]])




```python
# transpose
A.T
```




    tensor([[0, 2, 4],
            [1, 3, 5]])




```python
# High order tensors 
X = torch.arange(24).reshape(2, 3, 4)
X
```




    tensor([[[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11]],
    
            [[12, 13, 14, 15],
             [16, 17, 18, 19],
             [20, 21, 22, 23]]])




```python
# Assign a copy of matrix by allocating new memory
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()
A, A+B
```




    (tensor([[0., 1., 2.],
             [3., 4., 5.]]),
     tensor([[ 0.,  2.,  4.],
             [ 6.,  8., 10.]]))




```python
# The elementwise product of two matrices (Hadamard product)
A * B
```




    tensor([[ 0.,  1.,  4.],
            [ 9., 16., 25.]])




```python
# Sum over one axis: row and column dimensions 
A.shape, A.sum(axis = 0), A.sum(axis = 1)
```




    (torch.Size([2, 3]), tensor([3., 5., 7.]), tensor([ 3., 12.]))




```python
# computing the mean
A.mean(), A.sum() / A.numel()
```




    (tensor(2.5000), tensor(2.5000))




```python
# calculating the mean along specific axes
# numel() does not work on axis
A, A.mean(axis=0).shape, A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```




    (tensor([[0., 1., 2.],
             [3., 4., 5.]]),
     torch.Size([3]),
     tensor([1.5000, 2.5000, 3.5000]),
     tensor([1.5000, 2.5000, 3.5000]))




```python
# Non-Reduction reduction
A_sum = A.sum(axis = 1, keepdims=True)
A_sum, A_sum.shape
```




    (tensor([[ 3.],
             [12.]]),
     torch.Size([2, 1]))




```python
# cumulative sum of elements along axis=0 (row by row)
A, A.cumsum(axis=0)
```




    (tensor([[0., 1., 2.],
             [3., 4., 5.]]),
     tensor([[0., 1., 2.],
             [3., 5., 7.]]))




```python
# dot product of two vectors
x = torch.arange(3, dtype=torch.float32)
y = torch.ones(3, dtype=torch.float32)
x, y, torch.dot(x, y), x*y, torch.sum(x * y)
```




    (tensor([0., 1., 2.]),
     tensor([1., 1., 1.]),
     tensor(3.),
     tensor([0., 1., 2.]),
     tensor(3.))




```python
# Matrix-Vector Products
# mv : matrix - vector
A, A.shape, x, x.shape, torch.mv(A,x), A@x
```




    (tensor([[0., 1., 2.],
             [3., 4., 5.]]),
     torch.Size([2, 3]),
     tensor([0., 1., 2.]),
     torch.Size([3]),
     tensor([ 5., 14.]),
     tensor([ 5., 14.]))




```python
# Matrix-Vector-Matrix Products
A, x, A@x@A
```




    (tensor([[0., 1., 2.],
             [3., 4., 5.]]),
     tensor([0., 1., 2.]),
     tensor([42., 61., 80.]))




```python
# matrix-matrix multiplication
# mm : matrix-matrix 
B = torch.ones(3, 4)
A, B, A@B, torch.mm(A, B)
```




    (tensor([[0., 1., 2.],
             [3., 4., 5.]]),
     tensor([[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.]]),
     tensor([[ 3.,  3.,  3.,  3.],
             [12., 12., 12., 12.]]),
     tensor([[ 3.,  3.,  3.,  3.],
             [12., 12., 12., 12.]]))




```python
# Calculating the l2 norm (Euclidean distance)
u = torch.tensor([-5.3, 8])
torch.norm(u)
```




    tensor(9.5964)




```python
# Calculating the l2 norm (Manhattan distance)
torch.abs(u).sum()
```




    tensor(13.3000)




```python
# calculate the Frobenius norm of a matrix
# The Frobenius norm behaves as if it were an norm of a matrix-shaped vector
torch.norm(torch.ones(3, 8))
```




    tensor(4.8990)


