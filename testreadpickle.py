#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:21:24 2023

@author: wsb
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import pickle 

root = '/Users/wsb/Library/CloudStorage/OneDrive-SharedLibraries-UniversityofSouthampton/Phase retrieval with neural nets - Documents/nanoparticle Mie scattering project/mie Python code/wsb working'

with open("{}/rhys_int.pickle".format(root), "rb") as file:
    # Load the data from the file
    ints = pickle.load(file)
    
with open("{}/rhys_angles.pickle".format(root), "rb") as file:
    # Load the data from the file
    angles = pickle.load(file)
    
plt.figure()
plt.plot(angles[1,:],ints[1,:])
plt.show()

# could rewrite as a full pickle at this point, I guess
