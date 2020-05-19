#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:45:56 2020

@author: ar
"""
import sys
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import csv
import sys
import os
import matplotlib
import seaborn as sb

test = pd.read_csv('results_Ambiente1_exp3_ganma0.6000000000000001_epsilon1e-05_tempo2.7721365445190007.txt', header = None, sep = ",", engine='python')        
Rm = pd.read_csv('results_Ambiente1_exp3_ganma0.6000000000000001_epsilon1e-05_tempo2.7721365445190007.txt', header = None, sep = ",", engine='python')        
Rm = Rm.drop([3], axis=1) 
Rm = Rm.to_numpy()
#Rm = np.flip(Rm)
Vm = Rm[:,1]
Pm = Rm[:,2]

Vm = Vm.reshape(5,25)
#pi_grafico = Pm.reshape(Y,X)
heat_map = sb.heatmap(Vm)
plt.show()
plt.savefig(os.path.join("Img_ambiente_%s_ep_%s_g_%s.png" % (ambiente,epsilon,ganma)), bbox_inches='tight')



