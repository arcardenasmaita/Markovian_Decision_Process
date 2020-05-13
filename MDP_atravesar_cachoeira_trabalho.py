#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:01:19 2020

@author: ar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Criado: 10/04/2020
@Autor: ARCM 
@Tipo: Processo de Decisao de Markov MDP
@Algoritmo de Iteracao Valor (Value Iteration)
"""
import os
import sys
import numpy as np
from time import time 
sys.setrecursionlimit(10000)

### Classe MDP

class MDP(object):
    def __init__(self, So, G, X, Y, ambiente):        
        self.S = X*Y   # S = numero de estados
        self.So = So # So = estado inicial
        self.G = G   # G = estado objetivo
        self.X = X   # X = eixo x
        self.Y = Y   # Y = eixo y
        self.A = 4   # nro de acoes posiveis        
        self.T_norte = [] # Probabiliade de transicao para a acao Norte
        self.T_sul = []   # Probabiliade de transicao para a acao Sur
        self.T_leste = [] # Probabiliade de transicao para a acao Leste 
        self.T_oeste = [] # Probabiliade de transicao para a acao Oeste
        self.R = []       # matriz de custo
        self.ambiente = ambiente
        self.loadData()   # carregar os dados da probalidade e custo por acao
    def isEnd(self, state):        
        return state == self.G
    def states(self):
        return range(1, self.S+1)
    def actions(self):
        return range(0, self.A) # 0:norte, 1:sul, 2:leste, 3:oeste
    def R_action(self, state, action):              
        return self.R[state][action]
    def T(self, s_in, s_out, action):
        if action == 0:
            for tripla in self.T_norte:
                return float(tripla[2]) if (tripla[0] == s_in and tripla[1] == s_out) else 0.
        elif action == 1:
            for tripla in self.T_sul:
                return float(tripla[2]) if (tripla[0] == s_in and tripla[1] == s_out) else 0.
        elif action == 2:
            for tripla in self.T_leste:
                return float(tripla[2]) if (tripla[0] == s_in and tripla[1] == s_out) else 0.
        elif action == 3:
            for tripla in self.T_oeste:
                return float(tripla[2]) if (tripla[0] == s_in and tripla[1] == s_out) else 0.
    def loadData(self):        
        #file = open(self.actionfile+'/Action_Leste.txt')
        file = open('Ambiente1/Action_Norte.txt')
        for line in file:
            self.T_norte.append([int(float(line.strip().split()[0])),
                                 int(float(line.strip().split()[1])),
                                 float(line.strip().split()[2])])
        file.close()
        
        file = open('Ambiente1/Action_Sul.txt')
        for line in file:
            self.T_sul.append([int(float(line.strip().split()[0])),
                                 int(float(line.strip().split()[1])),
                                 float(line.strip().split()[2])])
        file.close()
        
        file = open('Ambiente1/Action_Leste.txt')
        for line in file:
            self.T_leste.append([int(float(line.strip().split()[0])),
                                 int(float(line.strip().split()[1])),
                                 float(line.strip().split()[2])])
        file.close()
        
        file = open('Ambiente1/Action_Oeste.txt')
        for line in file:
            self.T_oeste.append([int(float(line.strip().split()[0])),
                                 int(float(line.strip().split()[1])),
                                 float(line.strip().split()[2])])
        file.close()
        
        file = open('Ambiente1/Cost.txt')
        for line in file:
            self.R.append(int(float(line)))
        file.close()       
        
        
    def printSolution(self, V, pi):
        # | 0 | 1 | 2 | 3 | 4 | y=0
        # | 5 | 6 | 7 | 8 | 9 | y=1
        #  x=0 x=1 x=2 x=3 x=4
        s_river = np.reshape(V,(self.X,self.Y))
        polices = np.reshape(pi,(self.X,self.Y))        
        print("Valores otimos")
        print(s_river)
        print("Politica ótima")
        print(polices)
        pi_grafico = np.empty(self.S, dtype='str')
        for p in range(self.S):
            if pi[p] == 0.:
                pi_grafico[p] = '^'
            elif pi[p] == 1.:
                pi_grafico[p] = 'v'
            elif pi[p] == 2.:
                pi_grafico[p] = '>'
            elif pi[p] == 3.:
                pi_grafico[p] = '<'
                
        for s in self.states():
            print('(S:{:2}, V:{:5}, pi:{:5} {:2})'.format(s, V[s], pi[s], pi_grafico[s]))
        

### Algoritmos
        
def valueIteration(mdp, epsilon, ganma):
    V = np.zeros([mdp.S]) # almacenar o Voptimo[estado]    
    V_old = np.zeros([mdp.S])
    V[mdp.S-1] = 0.
    pi = np.zeros([mdp.S]) # politicas otimas
    res = np.inf
    
    Q = np.zeros([mdp.S,mdp.A])
    print('{:15} {:15} {:15}'.format('s', 'V(s)', 'pi(s)'))
    while res > epsilon:
        np.copyto(V_old, V) 
        for state in mdp.states():
            for action in mdp.actions():
                Q[state-1,action] = mdp.R[state-1]
                for sNext in mdp.states():
                    Q[state-1,action] = Q[state-1,action] + ganma*mdp.T(state,sNext,action)*V_old[sNext-1]
                #print ('*{:1} {:15}'.format(state, Q[state,action]))
            V[state-1] = np.max(Q, axis=1)[state-1] 
            pi[state-1] = np.argmax(Q, axis=1)[state-1]            
        # verificar a convergencia
        res = 0
        for s in mdp.states():
            dif = abs(V_old[s-1]-V[s-1])
            if dif > res:
                res = dif

    # escribir los resultados
    os.system('clear')
    
    #print('{:15} {:15} {:15}'.format('s', 'V(s)', 'pi(s)'))
    for state in mdp.states():
        print ('{:1} {:15} {:15}'.format(state, V[state-1], pi[state-1]))
    return V,pi

tempo_inicial_class = time()
mdp = MDP(So=1, G=9, X=5, Y=25, ambiente='Ambiente1')
tempo_final_class = time()
tempo_execucao_class = tempo_inicial_class - tempo_final_class
print ('O tempo de execucao da classe: ',tempo_execucao_class) #En segundos
tempo_inicial_vi = time()
V_opt, pi_opt = valueIteration(mdp, 0.000001, 1)
tempo_final_vi = time()
tempo_execucao_vi = tempo_inicial_vi - tempo_final_vi
print ('O tempo de execucao do VI: ',tempo_execucao_vi*60) #En min
mdp.printSolution(V_opt, pi_opt)
