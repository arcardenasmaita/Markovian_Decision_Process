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
X = 0           # número de colunas da matriz do mundo
Y = 0           # número de filas da matriz do mundo
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
Epsilon = 0.000001
Ganma = 0.8
Lao_t = 0
Vi_t = 0
Vi_converg = pd.DataFrame(columns=['it','error'])
Vi_calc = 0
#------------------------------------------
# Definição de classes
#------------------------------------------
class MDP(object):
    def __init__(self, So, G):        
        self.G = G
        self.So = So
        self.i = 0
        self.T_norte = pd.read_csv(Enviroment+'/Action_Norte.txt', header = None, sep = "   ", engine='python')        
        self.T_sul = pd.read_csv(Enviroment+'/Action_Sul.txt', header = None, sep = "   ", engine='python')        
        self.T_leste = pd.read_csv(Enviroment+'/Action_Leste.txt', header = None, sep = "   ", engine='python')        
        self.T_oeste = pd.read_csv(Enviroment+'/Action_Oeste.txt', header = None, sep = "   ",engine='python')        
        self.T_norte =np.array(T_norte)
        self.T_sul =np.array(T_sul)
        self.T_leste =np.array(T_leste)
        self.T_oeste =np.array(T_oeste)
        self.C_matrix = pd.read_csv(Enviroment+'/Cost.txt', header = None, engine='python')                
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
        if self.isEnd(state):
            c = 0
        else:
            c = -1.
            
        if action == 'N':
            for item in self.T_norte:
                #print(type(item))
                #print(item)
                if item[0] == state:
                    result.append((item[1], item[2], c))
        elif action == 'S':
            for item in self.T_sul:
                if item[0] == state:
                    result.append((item[1], item[2], c))
        elif action == 'L':
            for item in self.T_leste:
                if item[0] == state:
                    result.append((item[1], item[2], c))                
        elif action == 'O':
            for item in self.T_oeste:
                if item[0] == state:
                    result.append((item[1], item[2], c))
        return result
        i = i+1
        print(str(i))
    def discount(self):
        return 1.
    def states(self):
        return range(1, self.G+1)
    
#------------------------------------------
# Métodos para definir o mundo
#------------------------------------------
def loadData(enviroment, x, y, a, so, g):   
    global A, G, So, S, X, Y, Enviroment
    global T_norte, T_sul, T_leste, T_oeste, C_matrix, T
    os.system('clear')
    Enviroment = enviroment
    S = x*y
    X = x
    Y = y
    A = a 
    So = so
    G = g
    T_norte = pd.read_csv(Enviroment+'/Action_Norte.txt', header = None, sep = "   ", engine='python')        
    T_sul = pd.read_csv(Enviroment+'/Action_Sul.txt', header = None, sep = "   ", engine='python')        
    T_leste = pd.read_csv(Enviroment+'/Action_Leste.txt', header = None, sep = "   ", engine='python')        
    T_oeste = pd.read_csv(Enviroment+'/Action_Oeste.txt', header = None, sep = "   ",engine='python')        
    C_matrix = pd.read_csv(Enviroment+'/Cost.txt', header = None, engine='python')                
    T_norte.columns=('s_in', 's_out', 'prob')
    T_sul.columns=('s_in', 's_out', 'prob')
    T_leste.columns=('s_in', 's_out', 'prob')
    T_oeste.columns=('s_in', 's_out', 'prob')        

    
def printSolution(V, Pi, nr_exp):        
    # resultado dos parámetros
    print('*********************************************')
    print('  Ambiente: ' + Enviroment)
    print('  Experimento: ' + str(nr_exp))
    print('  Ganma: ' + str(Ganma))
    print('  Epsilon: '+ str(Epsilon))
    print('  Tempo VI: '+ str(Vi_t))
    #print('  Nro iteracoes: ' + str(Vi_converg.loc['it']))
    #print('  Erro converg: ' + str(Vi_converg.loc['error']))
    print('  Nro calculos de Q: ' + str(Vi_calc))
    print('*********************************************')        
    Vi_converg.to_csv(os.path.join("VI_%s_converg_amb_%s_ep_%s_g_%s_calc_%s" % (nr_exp,Enviroment,Epsilon,Ganma,str(Vi_calc))), index=False) 
    
    # resultados dos valores e politicas
    Vm = pd.DataFrame.from_dict(V.values())
    Pm = pd.DataFrame.from_dict(Pi.items())    
    Vm = Vm.to_numpy()
    Vm = Vm.reshape(X,Y)
    
    heat_mapV = sb.heatmap(Vm)
    figure1 = heat_mapV.get_figure()
    figure1.show()
    figure1.savefig(os.path.join("VI_%s_img_V_amb_%s_ep_%s_g_%s.png" % (nr_exp,Enviroment,Epsilon,Ganma)))
    
    # plot convergencia
    # x = Vi_converg.loc[:,'it']
    # y = Vi_converg.loc[:,'error']
    Vi_converg.plot(kind='line',x='it',y='error', color='red')
    plt.show()

    # plt.title("Experimento en ambiente %s, epsilon %s e ganma %s" % (Enviroment,Epsilon,Ganma))
    # plt.ylabel('Error')
    # plt.xlabel('Iteracoes')
    # plt.plot(x,y, lw=2)        
    # plt.savefig(fname = "VI_%s_img_converg_amb_%s_ep_%s_g_%s.png"%(nr_exp,Enviroment,Epsilon,Ganma), bbox_inches='tight')
    print(Pm)

#------------------------------------------
#------------------------------------------
# ******* ALGORITMOS *********
#------------------------------------------
#------------------------------------------

def valueIteration(mdp):
    global Vi_t, Vi_calc, Vi_converg
    Vi_t_ini = time()    
    
    # inicializar valores
    V = {} # estado ->Voptimo[estado]
    
    for state in mdp.states():
        V[state] = 0.    
    def Q(state, action):
        q_temp = []
        #Vi_calc = 0
        for newState, prob, reward in mdp.succesorProbReward(state, action):
            print('S=%s A=%s nS=%s P=%s R=%s' %(state, action,newState, prob,reward))
            print(str(prob*(reward + mdp.discount()*V[newState])))
            q_temp.append(prob*(reward + mdp.discount()*V[newState]))
            #Vi_calc = Vi_calc+ 1
            
        return sum(q_temp)
        # return sum(prob*(reward + mdp.discount()*V[newState]) \
        #            for newState, prob, reward in mdp.succesorProbReward(state, action))
    Vi_it = 0
    error = 10000000000
    while True:
        Vi_converg.append([Vi_it,error])
        Vi_it = Vi_it+1
        # calcular los nuevos valores 'newV' dados los anteriores valores 'V'
        newV = {}
        pi = {}
        for state in mdp.states():
            if mdp.isEnd(state):
                newV[state] = 0.
                pi[state] = 'none'
            else:
                newV[state] = max(Q(state,action) for action in mdp.actions(state))        
                # recuperar argumentos que maximizan la función Q (argmax)
                pi[state] = max((Q(state, action), action) for action in mdp.actions(state))
                
        # verificar la convergencia
        # res = 0
        # for state in mdp.states():
        #     dif = abs(V[state]-newV[state])
        #     if max(dif) < Epsilon:
        #         Vi_converg.append([Vi_it,max(dif)])
        #         Vi_t_fin = time()    
        #         Vi_t = (Vi_t_fin-Vi_t_ini)/360
        #         break
        
        error = max(abs(V[state]-newV[state]) for state in mdp.states()) 
        print(error)
        if error < Epsilon:
            Vi_t_fin = time()    
            Vi_t = (Vi_t_fin-Vi_t_ini)/360
            break
        V = newV        
    return V, pi     
        

    
#------------------------------------------
# Main()
#------------------------------------------

def main():    
    # teste
    #loadData(enviroment ='Ambiente1', x = 5, y = 25, a = 4, so = 1, g = 125)    
    loadData(enviroment ='Ambiente0v', x = 2, y = 5, a = 4, so = 6, g = 10)        
    mdp = MDP(So, G)    
    # print(mdp.succesorProbReward(1.0,'N'))
    # print(mdp.succesorProbReward(1.0,'S'))
    # print(mdp.succesorProbReward(1.0,'L'))
    # print(mdp.succesorProbReward(1.0,'O'))
    V,pi = valueIteration(mdp)
    printSolution(V,pi,1)
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