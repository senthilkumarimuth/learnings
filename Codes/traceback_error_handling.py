# -*- coding: utf-8 -*-
"""
To stop the program throw error

"""


import traceback

try:
    1/0
except:
    traceback.print_exc()

