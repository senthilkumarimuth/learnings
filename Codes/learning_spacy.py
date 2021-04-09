# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:16:30 2020

@author: senthilkumar.m02
"""


import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a sentence.")