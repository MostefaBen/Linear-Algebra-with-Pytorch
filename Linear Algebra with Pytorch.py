#!/usr/bin/env python
# coding: utf-8

# # Linear Algebra with Pytorch

# In[4]:


# import libraries
import torch
import numpy as np


# In[5]:


# matrix
A = torch.arange(6).reshape(3, 2)
A


# In[6]:


# transpose
A.T


# In[7]:


# High order tensors 
X = torch.arange(24).reshape(2, 3, 4)
X


# In[8]:


# Assign a copy of matrix by allocating new memory
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()
A, A+B


# In[9]:


# The elementwise product of two matrices (Hadamard product)
A * B


# In[10]:


# Sum over one axis: row and column dimensions 
A.shape, A.sum(axis = 0), A.sum(axis = 1)


# In[11]:


# computing the mean
A.mean(), A.sum() / A.numel()


# In[12]:


# calculating the mean along specific axes
# numel() does not work on axis
A, A.mean(axis=0).shape, A.mean(axis=0), A.sum(axis=0) / A.shape[0]


# In[13]:


# Non-Reduction reduction
A_sum = A.sum(axis = 1, keepdims=True)
A_sum, A_sum.shape


# In[14]:


# cumulative sum of elements along axis=0 (row by row)
A, A.cumsum(axis=0)


# In[15]:


# dot product of two vectors
x = torch.arange(3, dtype=torch.float32)
y = torch.ones(3, dtype=torch.float32)
x, y, torch.dot(x, y), x*y, torch.sum(x * y)


# In[16]:


# Matrix-Vector Products
# mv : matrix - vector
A, A.shape, x, x.shape, torch.mv(A,x), A@x


# In[17]:


# Matrix-Vector-Matrix Products
A, x, A@x@A


# In[18]:


# matrix-matrix multiplication
# mm : matrix-matrix 
B = torch.ones(3, 4)
A, B, A@B, torch.mm(A, B)


# In[19]:


# Calculating the l2 norm (Euclidean distance)
u = torch.tensor([-5.3, 8])
torch.norm(u)


# In[20]:


# Calculating the l2 norm (Manhattan distance)
torch.abs(u).sum()


# In[21]:


# calculate the Frobenius norm of a matrix
# The Frobenius norm behaves as if it were an norm of a matrix-shaped vector
torch.norm(torch.ones(3, 8))

