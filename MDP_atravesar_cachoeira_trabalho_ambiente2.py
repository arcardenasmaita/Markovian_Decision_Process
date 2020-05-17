#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Criado: 10/04/2020
@Autor: ARCM 
@Tipo: Processo de Decisao de Markov MDP
@Algoritmo de Iteracao Valor (Value Iteration)
"""
import sys
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import csv
import sys
import os

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
        self.T_norte = pd.DataFrame() # Probabiliade de transicao para a acao Norte
        self.T_sul = pd.DataFrame()   # Probabiliade de transicao para a acao Sur
        self.T_leste = pd.DataFrame() # Probabiliade de transicao para a acao Leste 
        self.T_oeste = pd.DataFrame() # Probabiliade de transicao para a acao Oeste
        self.R_matrix = pd.DataFrame()       # matriz de custo
        self.ambiente = ambiente
        self.loadData()   # carregar os dados da probalidade e custo por acao
    def isEnd(self, state):        
        return state == self.G
    def states(self):
        return range(1, self.S+1)
    def actions(self):
        return range(0, self.A) # 0:norte, 1:sul, 2:leste, 3:oeste
    def R(self, state, action):              
        return float(self.R_matrix[0][state])
    def T(self, s_in, s_out, action):                
        if action == 0:
            return float(self.T_norte[(self.T_norte['s_in'] == s_in) & (self.T_norte['s_out'] == s_out)]["prob"]) if not self.T_norte[(self.T_norte['s_in'] == s_in) & (self.T_norte['s_out'] == s_out)].empty else 0
        elif action == 1:
            return float(self.T_sul[(self.T_sul['s_in'] == s_in) & (self.T_sul['s_out'] == s_out)]["prob"]) if not self.T_sul[(self.T_sul['s_in'] == s_in) & (self.T_sul['s_out'] == s_out)].empty else 0    
        elif action == 2:
            return float(self.T_leste[(self.T_leste['s_in'] == s_in) & (self.T_leste['s_out'] == s_out)]["prob"]) if not self.T_leste[(self.T_leste['s_in'] == s_in) & (self.T_leste['s_out'] == s_out)].empty else 0
        elif action == 3:
            return float(self.T_oeste[(self.T_oeste['s_in'] == s_in) & (self.T_oeste['s_out'] == s_out)]["prob"]) if not self.T_oeste[(self.T_oeste['s_in'] == s_in) & (self.T_oeste['s_out'] == s_out)].empty else 0
        else:
            return 0
    def loadData(self):                
        #self.T_norte = pd.read_csv('Ambiente1/Action_Norte.txt', header = None, sep = "   ")        
        #self.T_sul = pd.read_csv('Ambiente1/Action_Sul.txt', header = None, sep = "   ")        
        #self.T_leste = pd.read_csv('Ambiente1/Action_Leste.txt', header = None, sep = "   ")        
        #self.T_oeste = pd.read_csv('Ambiente1/Action_Oeste.txt', header = None, sep = "   ")        
        #self.R_matrix = pd.read_csv('Ambiente1/Cost.txt', header = None)        
        
        self.T_norte = pd.read_csv(self.ambiente+'/Action_Norte.txt', header = None, sep = "   ", engine='python')        
        self.T_sul = pd.read_csv(self.ambiente+'/Action_Sul.txt', header = None, sep = "   ", engine='python')        
        self.T_leste = pd.read_csv(self.ambiente+'/Action_Leste.txt', header = None, sep = "   ", engine='python')        
        self.T_oeste = pd.read_csv(self.ambiente+'/Action_Oeste.txt', header = None, sep = "   ",engine='python')        
        self.R_matrix = pd.read_csv(self.ambiente+'/Cost.txt', header = None, engine='python')        
        
        self.T_norte.columns=('s_in', 's_out', 'prob')
        self.T_sul.columns=('s_in', 's_out', 'prob')
        self.T_leste.columns=('s_in', 's_out', 'prob')
        self.T_oeste.columns=('s_in', 's_out', 'prob')
    def printSolution(self, V, pi):       
        #s_river = np.reshape(V,(self.X,self.Y))
        #polices = np.reshape(pi,(self.X,self.Y))        
        #print("Valores otimos")
        #print(s_river)
        #print("Politica ótima")
        #print(polices)
        #pi_grafico = np.empty(self.S, dtype='str')
        #for p in pi:
        #   if p == 0.:
        #        pi_grafico[p.index] = '^'
        #    elif p == 1.:
        #        pi_grafico[p.index] = 'v'
        #    elif p == 2.:
        #        pi_grafico[p.index] = '>'
        #    elif p == 3.:
        #        pi_grafico[p.index] = '<'
        
        results = pd.DataFrame(columns=['s', 'V', 'pi', 'pi_grafico'])        
        print('{:2} {:5} {:5}'.format('s', 'V(s)', 'pi(s)'))       
        for s in range(0,self.S):
            print('(S:{:2}, V:{:5}, pi:{:5})'.format(s+1, V[s], pi[s]))
            results = results.append({'s': s+1,
                                     'V': V[s], 
                                     'pi': pi[s]}, ignore_index=True)    
                
        return results
        

### Algoritmos
        
def valueIteration(mdp, epsilon, ganma):
    V = np.zeros([mdp.S]) # almacenar o Voptimo[estado]    
    V_old = np.zeros([mdp.S])
    V[mdp.S-1] = 0.
    pi = np.zeros([mdp.S]) # politicas otimas
    res = np.inf
    Q = np.zeros([mdp.S,mdp.A])
    k = 0
   
    converg = pd.DataFrame(columns=['Iteracoes', 'Erro','q'])
    while res > epsilon:                
        q = 0
        k=k+1
        np.copyto(V_old, V) 
        for state in mdp.states():
            for action in mdp.actions():
                Q[state-1,action] = (-1)*mdp.R(state-1,action)
                for sNext in mdp.states():
                    Q[state-1,action] = Q[state-1,action] + ganma*mdp.T(state,sNext,action)*V_old[sNext-1]                
                    q = q+1
            V[state-1] = np.max(Q, axis=1)[state-1] 
            pi[state-1] = np.argmax(Q, axis=1)[state-1]            
        # verificar a convergencia
        res = 0
        for s in mdp.states():
            dif = abs(V_old[s-1]-V[s-1])
            if dif > res:
                res = dif
        converg.loc[k] = [k,res,q]
    return V,pi,converg

### Experimentos
def executarExperimentosVI(mdpObject,ambiente,nro_exp):
    ganma =  0.9
    epsilon = 0.01
    fig, axs = plt.subplots(nro_exp, 1, sharex=True)
    
    for i in range(1,nro_exp+1):
        tempo_inicial_vi = time()
        V_opt, pi_opt, converg = valueIteration(mdp, epsilon, ganma)
        tempo_final_vi = time()    
        tempo_execucao_vi = tempo_final_vi-tempo_inicial_vi
        print('*********************************************')
        print('  Ambiente: ' + ambiente)
        print('  Experimento: ' + nro_exp)
        print('  Ganma: ' + str(ganma))
        print('  Epsilon: '+ str(epsilon))
        print('  Tempo VI: '+ str(tempo_execucao_vi/360))
        print('  Nro iteracoes: ' + str(np.max(converg.loc[:,'error'])))
        print('  Nro calculos de Q: ' + str(np.max(converg.loc[:,'q'])))
        print('*********************************************')        
        # mostrar solucao e salvar resultados em arquivo        
        #results = mdp.printSolution(V_opt, pi_opt)
        # converg = pd.DataFrame(columns=['Iteracoes', 'Erro','q'])
        # converg.loc[9] = [2,2,3]
        # converg.loc[10] = [3,4,4]
        # converg.loc[12] = [4,6,4]
        
        # plot figuras
        axs[i-1].plot(converg.loc[:,'Iteracoes'], converg.loc[:,'Erro'], lw=2)        
        axs[i-1].set_title('Experimento'+str(i))
        
        # salvar resultados
        results.to_csv('res_'+ambiente+'_exp_'+str(i+1)+'_g_'+str(ganma)+'_e_'+str(epsilon)+'_t'+str(tempo_execucao_vi/360)+'_it_'+str(nr_iteration)+'_calq_'+str(nr_calculateq)+'.txt', index=False) 
        
        # variar os parámetros
        ganma = ganma - 0.2
        epsilon = epsilon*0.01      
    
    for ax in axs.flat:
        ax.set(xlabel='Iteracoes', ylabel='Erro')
        ax.label_outer()
        
    plt.savefig(os.path.join("ambiente_%s_ep_%s_g_%s.png" % (ambiente,epsilon,ganma)), bbox_inches='tight')


mdp = MDP(So=1, G=125, X=5, Y=25, ambiente='Ambiente1')
executarExperimentosVI(mdpObject = mdp, ambiente = 'Ambiente1', nro_exp = 3)

mdp = MDP(So=1, G=2000, X=20, Y=100, ambiente='Ambiente2')
executarExperimentosVI(mdpObject = mdp, ambiente = 'Ambiente2', nro_exp = 2)

mdp = MDP(So=1, G=12500, X=50, Y=250, ambiente='Ambiente3')
executarExperimentosVI(mdpObject = mdp, ambiente = 'Ambiente3', nro_exp = 2)
