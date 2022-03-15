# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:40:17 2020

Description: Findes line by line profiling for a function
             The profiler's metric is of inclusive of all runs, not metric per run
@author: senthilkumar.m02
"""

import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)

@profile
def is_addable(l, t):
    '''jh'''
    for i, n in enumerate(l):
        for m in l[i:]:
            if n + m == t:
                return True

    return False

assert is_addable(range(20), 25) == True
assert is_addable(range(20), 40) == False

