# -*- coding: utf-8 -*-
"""
Created on Fri May 15 21:56:42 2020

@author: senthilkumar.m02

Run using below command:
    
python -m memory_profiler line_by_line_memoryprofiler.py    
"""

@profile
def my_func():
     a = [1] * (10 ** 6)
     b = [2] * (2 * 10 ** 7)
     del b
     return a

my_func()