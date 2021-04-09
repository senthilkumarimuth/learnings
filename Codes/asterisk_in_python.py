# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 19:19:03 2020

@author: senthilkumar.m02
"""

#1. Unpacking a tuple

#sorting list based on length

my_list = ['senthil','kumar','M','Somebody']
len_list = []
for i in my_list:
    len_list.append((len(i),i))

len_list.sort()
a =list(zip(*len_list))[1]   #<------here

print(list(a))

#2. pass arguments to a function
from functools import reduce

def add(a,b,*arg): #<------here
    arg=[a,b,*arg]
    print(arg)
    result_temp  = reduce((lambda x,y:x+y),arg)
    print(result_temp)
    return result_temp

add(2,3,5,10,10,70)#should return 100