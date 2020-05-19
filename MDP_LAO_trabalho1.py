#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Autor: ARCM 
@Tipo: Proceso de Decision Markoviano MDP
@Definicion:
 * Proceso de Decision Markoviano *
Algoritmo de Iteracion Valor (Value Iteration)
"""

#------------------------------------------
# Librerias
#------------------------------------------
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

#------------------------------------------
# Variáveis globais
#------------------------------------------
A = 4           # nro de acoes posiveis        
S = 0           # nro total de estados
So = 0          # estado inicial
G = 0           # estado objetivo
x = 0           # número de colunas da matriz do mundo
y = 0           # número de filas da matriz do mundo
Enviroment = '' # ambiente para rodar os experimentos
T_norte = pd.DataFrame() # Probabiliade de transicao para a acao Norte
T_sul = pd.DataFrame()   # Probabiliade de transicao para a acao Sur
T_leste = pd.DataFrame() # Probabiliade de transicao para a acao Leste 
T_oeste = pd.DataFrame() # Probabiliade de transicao para a acao Oeste
C_matrix = pd.DataFrame()# matriz de custo        
G_So_I = set()           # conjunto de todos os nodos internos do hipergrafo G_So
G_So_F = set()           # conjunto de todos os nodos fronteira do hipergrafo G_So
G_So_v_I = set()         # conjunto de todos os nodos internos do hipergrafo G_So com a função V(s)
G_So_v_F = set()         # conjunto de todos os nodos fronteira do hipergrafo G_So com a função V(s)
V = pd.DataFrame()       # diccionario de valores para cada tupla (s_n, Vs_n)
Epsilon = 0.001
Ganma = 1
T = {}
Enviroment ='Ambiente1'
X = 5
Y = 25
A = 4
S=125
So = 1
G = 125

# Enviroment ='Ambiente0'
# X = 2
# Y = 5
# A = 4
# S = 10
# So = 1
# G = 10

#------------------------------------------
# Métodos para definir o mundo
#------------------------------------------
def loadData(Enviroment_, X_, Y_, A_, So_, G_):   
    global A, G, So, S, x, y, Enviroment
    global T_norte, T_sul, T_leste, T_oeste, C_matrix, T
    Enviroment = Enviroment_
    S = X_*Y_
    x = X_
    y = Y_
    A = A_ 
    So = So_
    G = G_
    T_norte = pd.read_csv(Enviroment+'/Action_Norte.txt', header = None, sep = "   ", engine='python')        
    T_sul = pd.read_csv(Enviroment+'/Action_Sul.txt', header = None, sep = "   ", engine='python')        
    T_leste = pd.read_csv(Enviroment+'/Action_Leste.txt', header = None, sep = "   ", engine='python')        
    T_oeste = pd.read_csv(Enviroment+'/Action_Oeste.txt', header = None, sep = "   ",engine='python')        
    C_matrix = pd.read_csv(Enviroment+'/Cost.txt', header = None, engine='python')                
    T_norte.columns=('s_in', 's_out', 'prob')
    T_sul.columns=('s_in', 's_out', 'prob')
    T_leste.columns=('s_in', 's_out', 'prob')
    T_oeste.columns=('s_in', 's_out', 'prob')        

    
# def printSolution(self, Rm,X,Y):    
#     Rm = Rm.drop([3], axis=1) 
#     Rm = Rm.to_numpy()
#     Vm = Rm[:,1]
#     #Pm = Rm[:,2]
#     Vm = Vm.reshape(X,Y)
#     heat_map = sb.heatmap(Vm)
#     figure = heat_map.get_figure()
#     figure.show()
#     figure.savefig(os.path.join("Img_ambiente_%s_ep_%s_g_%s.png" % (Enviroment,epsilon,ganma)))

#------------------------------------------
# Definição de classes
#------------------------------------------
class MDP(object):
    def __init__(self, So, G):        
        self.G = G
        self.So = So
        self.T_norte = pd.read_csv(Enviroment+'/Action_Norte.txt', header = None, sep = "   ", engine='python')        
        self.T_sul = pd.read_csv(Enviroment+'/Action_Sul.txt', header = None, sep = "   ", engine='python')        
        self.T_leste = pd.read_csv(Enviroment+'/Action_Leste.txt', header = None, sep = "   ", engine='python')        
        self.T_oeste = pd.read_csv(Enviroment+'/Action_Oeste.txt', header = None, sep = "   ",engine='python')        
        self.T_norte =np.array(T_norte)
        self.T_sul =np.array(T_sul)
        self.T_leste =np.array(T_leste)
        self.T_oeste =np.array(T_oeste)
        # self.C_matrix = pd.read_csv(Enviroment+'/Cost.txt', header = None, engine='python')                
        # self.T_norte.columns=('s_in', 's_out', 'prob')
        # self.T_sul.columns=('s_in', 's_out', 'prob')
        # self.T_leste.columns=('s_in', 's_out', 'prob')
        # self.T_oeste.columns=('s_in', 's_out', 'prob')   
    def startState(self):        
        return So
    def isEnd(self, state):        
        return state == self.G
    def actions(self, state):
        # devuelve una lista de aciones validas
        result = []
        # result.append('N')
        # result.append('S')
        # result.append('L')
        # result.append('O')
        result = ['N','S','L','O'] 
        # if state+1 <= self.N:
        #     result.append('walk')
        # if state*2 <= self.N:
        #     result.append('tram')
        return result

    def succesorProbReward(self, state, action):
        global i
        # retorna uma lista da tripla (novoEstado, probabilidade, recompensa)
        # estado = s, accion = a, nuevoEstado = s'
        # probabilidad = T(s, a, s'), recompensa = Reward(s, a, s')        
        result = [] 
        if action == 'N':
            for item in self.T_norte:
                #print(type(item))
                #print(item)
                if item[0] == state:
                    result.append((item[1], item[2], -1.))
        elif action == 'S':
            for item in self.T_sul:
                if item[0] == state:
                    result.append((item[1], item[2], -1.))
        elif action == 'L':
            for item in self.T_leste:
                if item[0] == state:
                    result.append((item[1], item[2], -1.))                
        elif action == 'O':
            for item in self.T_oeste:
                if item[0] == state:
                    result.append((item[1], item[2], -1.))
        return result
        # if action == 'N':
        #     for item in self.T_norte.itertuples():
        #         if item[0] == state:
        #             result.append((item[1], item[2], -1.))
        # elif action == 'S':
        #     for item in self.T_sul.itertuples():
        #         if item[0] == state:
        #             result.append((item[1], item[2], -1.))
        # elif action == 'L':
        #     for item in self.T_leste.itertuples():
        #         if item[0] == state:
        #             result.append((item[1], item[2], -1.))                
        # else:
        #     for item in self.T_oeste.itertuples():
        #         if item[0] == state:
        #             result.append((item[1], item[2], -1.))
        # return result
        # if action == 'tram':
        #     result.append((state*2, 0.5, -2.))
        #     result.append((state, 0.5, -2.))
        # i = i+1
        #print(i)
    def discount(self):
        return 1.
    def states(self):
        return range(1, self.G+1)
    
### Algoritmos

def valueIteration(mdp):
    # inicializar valores
    V = {} # estado ->Voptimo[estado]
    for state in mdp.states():
        V[state] = 0.    
    def Q(state, action):
        # q_temp = []
        # for item in mdp.succesorProbReward(state, action):
        #     q_temp.append(item[1]*(item[2] + mdp.discount()*V[item[0]]))
        #     print(newState,' ', prob,' ', reward)
        # return sum(q_temp)
        return sum(prob*(reward + mdp.discount()*V[newState]) \
                   for newState, prob, reward in mdp.succesorProbReward(state, action))
    while True:
        # calcular los nuevos valores 'newV' dados los anteriores valores 'V'
        newV = {}
        for state in mdp.states():
            if mdp.isEnd(state):
                newV[state] = 0.
            else:
                newV[state] = max(Q(state,action) for action in mdp.actions(state))        
        # verificar la convergencia
        if max(abs(V[state]-newV[state]) for state in mdp.states()) < Epsilon:
            break
        V = newV
        
        # leer la politica
        pi = {}
        for state in mdp.states():
            if mdp.isEnd(state):
                pi[state] = 'none'
            else:
                # recuperar argumentos que maximizan la función Q (argmax)
                pi[state] = max((Q(state, action), action) for action in mdp.actions(state))[1] 
        
        # escribir los resultados
        os.system('clear')
        
        print('{:15} {:15} {:15}'.format('s', 'V(s)', 'pi(s)'))
        for state in mdp.states():
            print ('{:15} {:15} {:15}'.format(state, V[state], pi[state]), sep="  ")
        #input()

    
#------------------------------------------
# Main()
#------------------------------------------

def main():    
    # teste
    loadData(Enviroment_ ='Ambiente1', X_ = 5, Y_ = 25, A_ = 4, So_ = 1, G_ = 125)    
    mdp = MDP(So, G)    
    # print(mdp.succesorProbReward(1.0,'N'))
    # print(mdp.succesorProbReward(1.0,'S'))
    # print(mdp.succesorProbReward(1.0,'L'))
    # print(mdp.succesorProbReward(1.0,'O'))
    valueIteration(mdp)
     
    #LAO(ganma_ =  1, epsilon_ = 0.001)
    #executarExperimentosLAO(mdpObject = lao, enviroment = 'Ambiente1', nro_exp = 3)
    
    #mdp = LAO(So=1, G=125, X=5, Y=25, enviroment='Ambiente1')
    #executarExperimentosVI(mdpObject = mdp, enviroment = 'Ambiente1', nro_exp = 3)
    
    #mdp = LAO(So=1, G=2000, X=20, Y=100, enviroment='Ambiente2')
    #executarExperimentosVI(mdpObject = mdp, enviroment = 'Ambiente2', nro_exp = 3)
    
    #mdp = LAO(So=1, G=12500, X=50, Y=250, enviroment='Ambiente3')
    #executarExperimentosVI(mdpObject = mdp, enviroment = 'Ambiente3', nro_exp = 3)

if __name__ == "__main__":
    main()