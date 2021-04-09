# -*- coding: utf-8 -*-
"""
Created on Fri May 22 21:12:52 2020

@author: senthilkumar.m02
"""


from contextlib import contextmanager
@contextmanager
def handler():
    # Put here what would ordinarily go in the `__enter__` method
    # In this case, there's nothing to do
    try:
        yield # You can return something if you want, that gets picked up in the 'as'
    except Exception as e:
        print(e)
    finally:
        pass

with handler():
    name=1/0 #to raise an exception
    print(name)