# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 18:05:38 2020

@author: senthilkumar.m02
"""

# Normal copy via assignment operator

a = [1,2,3,4,5] 
b = a 
id(a) == id(b) # Returns True

#Shallow copy

import copy

a = [1,2,3,4,5]
b = copy.copy(a)
id(a)==id(b) # Returns False

#exception for nested array

a=[[1,2,3],[4,5,6]]
b= copy.copy(a)
id(a)==id(b)
False
b[1][0]=8
b

#Deep copy

import copy

a=[[1,2,3],[4,5,6]]
b= copy.deepcopy(a)
id(a)==id(b)
False
b[1][0]=8
b