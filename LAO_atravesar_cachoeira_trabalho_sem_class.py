#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Criado: 10/04/2020
@Autor: ARCM 
@Tipo: Processo de Decisao de Markov MDP
@Algoritmo LAO*
"""
#------------------------------------------
# Librerias
#------------------------------------------
#import sys
import numpy as np
import pandas as pd
from time import time 

#------------------------------------------
# Variáveis globais
#------------------------------------------

A = 4           # nro de acoes posiveis        
S = 0           # nro total de estados
So = 0          # estado inicial
G = 0           # estado objetivo
x = 0           # número de colunas da matriz do mundo
y = 0           # número de filas da matriz do mundo
enviroment = '' # ambiente para rodar os experimentos
T_norte = pd.DataFrame() # Probabiliade de transicao para a acao Norte
T_sul = pd.DataFrame()   # Probabiliade de transicao para a acao Sur
T_leste = pd.DataFrame() # Probabiliade de transicao para a acao Leste 
T_oeste = pd.DataFrame() # Probabiliade de transicao para a acao Oeste
C_matrix = pd.DataFrame()# matriz de custo        
G_So_I = set()           # conjunto de todos os nodos internos do hipergrafo G_So
G_So_F = set()           # conjunto de todos os nodos fronteira do hipergrafo G_So
G_So_v_I = set()         # conjunto de todos os nodos internos do hipergrafo G_So com a função V(s)
G_So_v_F = set()         # conjunto de todos os nodos fronteira do hipergrafo G_So com a função V(s)
V = {}                   # diccionario de valores para cada tupla (s_n, Vs_n)

#------------------------------------------
# Métodos para definir o mundo
#------------------------------------------

def loadData(enviroment_, X_, Y_, A_, So_, G_):   
    global A, G, So, S, x, y, enviroment
    global T_norte, T_sul, T_leste, T_oeste, C_matrix
    enviroment = enviroment_
    S = X_*Y_
    x = X_
    y = Y_
    A = A_ 
    So = So_
    G = G_
    T_norte = pd.read_csv(enviroment+'/Action_Norte.txt', header = None, sep = "   ", engine='python')        
    T_sul = pd.read_csv(enviroment+'/Action_Sul.txt', header = None, sep = "   ", engine='python')        
    T_leste = pd.read_csv(enviroment+'/Action_Leste.txt', header = None, sep = "   ", engine='python')        
    T_oeste = pd.read_csv(enviroment+'/Action_Oeste.txt', header = None, sep = "   ",engine='python')        
    C_matrix = pd.read_csv(enviroment+'/Cost.txt', header = None, engine='python')                
    T_norte.columns=('s_in', 's_out', 'prob')
    T_sul.columns=('s_in', 's_out', 'prob')
    T_leste.columns=('s_in', 's_out', 'prob')
    T_oeste.columns=('s_in', 's_out', 'prob')
    
def printSolution(V, pi):               
    return "results"
    
def actions(A):
    return range(1, A+1) # 1:norte, 2:sul, 3:leste, 4:oeste

def T(s_in, s_out, action):                
    if action == 1:
        return float(T_norte[(T_norte['s_in'] == s_in) & (T_norte['s_out'] == s_out)]["prob"]) if not T_norte[(T_norte['s_in'] == s_in) & (T_norte['s_out'] == s_out)].empty else 0
    elif action == 2:
        return float(T_sul[(T_sul['s_in'] == s_in) & (T_sul['s_out'] == s_out)]["prob"]) if not T_sul[(T_sul['s_in'] == s_in) & (T_sul['s_out'] == s_out)].empty else 0    
    elif action == 3:
        return float(T_leste[(T_leste['s_in'] == s_in) & (T_leste['s_out'] == s_out)]["prob"]) if not T_leste[(T_leste['s_in'] == s_in) & (T_leste['s_out'] == s_out)].empty else 0
    elif action == 4:
        return float(T_oeste[(T_oeste['s_in'] == s_in) & (T_oeste['s_out'] == s_out)]["prob"]) if not T_oeste[(T_oeste['s_in'] == s_in) & (T_oeste['s_out'] == s_out)].empty else 0
    else:
        return 0
    
def states(S):
    return range(1, S+1)

def C(state):              
    if isGoal(state):
        return 0
    else:
        return -1
    
#------------------------------------------
# Métodos para definir o hipergrafo
#------------------------------------------
def expand(node):
    # buscar nodos siguientes ao node
    
    return ''

def isGoal(node):
    return node == G

def nodes(I, F):
    return I.union(F)

def initializeGraph():
    V = {So:0}     
    V[So] = heuristic(So) 
    G_So_F.add(So)
    G_So_v_F.add(So)    
   
def heuristic(state): # H(s)
    return 0

def accumulated_C(): # C(s)
    return 0

def function_V(): # V(s)
    #H(s) ≤ V (s)
    return 0

def listAdjacent(s):
    return {7,1}

#------------------------------------------
# Algoritmo Value Iteration
#------------------------------------------

def valueIteration(S_list, epsilon, ganma):
    V = pd.DataFrame(0, index = range(0,len(S_list)), columns = ['S', 'V']) # almacenar o Voptimo[estado]        
    V_old = pd.DataFrame(0, index = range(0,len(S_list)), columns = ['S',''])
    #V[len(S_list)-1] = 0.
    pi = pd.DataFrame(0, index=range(0,len(S_list)), columns=['pi']) # politicas otimas
    res = np.inf
    Q = pd.DataFrame(0, index=range(0,len(S_list)), columns = range(1,A+1))
        
    while res > epsilon:
        V_old = V.copy()
        for index_s in range(0,len(S_list)):
            for action in actions(A):
                Q.loc[index_s,action] = C(S_list[index_s])
                for sNext in range(0,len(S_list)):
                    #print(index_s,action)
                    Q.loc[index_s,action] = Q.loc[index_s,action] + ganma*T(S_list[index_s], S_list[sNext],action)*V_old.loc[sNext,'V']
            V.loc[index_s,'V'] = np.max(Q, axis=1)[index_s] 
            pi.loc[index_s,'pi'] = Q.idxmax(axis=1)[index_s]        
        # verificar a convergencia
        res = 0
        for s in range(0, len(S_list)):
            dif = abs(V_old.loc[s,'V']-V.loc[s,'V'])
            if dif > res:
                res = dif
    return V,pi

#------------------------------------------
# Algoritmo LAO*
#------------------------------------------

def LAO(epsilon, ganma):    
    global G_So_I, G_So_F, G_So_v_I, G_So_v_F, V
    
    # 1. inicializa uma ´arvore com o n´o raiz s0    
    initializeGraph()
    pi_So_v = set()
    Z = set()   
    G_So = G_So_I.union(G_So_F)
                
    # 2. repete enquanto existam nos na fronteira e G_So_v
    nodesForVisit = G_So_F.intersection(G_So_v_F)
    s = 0
    while len(nodesForVisit) > 0 and not isGoal(s):
        # a) escolha um n´o folha para expandir
        s = nodesForVisit.pop()
        # b) retira s do conjunto de nós fronteira
        G_So_F.remove(s)
        # c) adiciona s ao conjunto de nós internos
        G_So_I.add(s)
        # d) expandir, adicionar a F os adjacentes com T>0
        adj = listAdjacent(s)
        
        for node_adj in adj:
            G_So_F.add(node_adj)
        
        # e) atualiza estados em G_So
        G_So = G_So_I.union(G_So_F)
        
        # f) todos os estados que podem alcancar G, melhores acoes
        
        # g) atualiza V para todo S em Z, aplicar VI
        # quem que está em G_So_v e que consegui chegar a s
        V_So, pi_So = valueIteration(list(G_So), epsilon, ganma)
        V.setdefault(V_So)
        # h) reconstroi G_Vs0
        
        nodesForVisit = G_So_F.intersection(G_So_v_F)
    
    # 3. retorne a política parcial desde So    
    return pi_So_v

#------------------------------------------
# Métodos para executar os experimentos
#------------------------------------------

def executarExperimentosLAO(laoObject,enviroment,nro_exp):
    ganma =  1
    epsilon = 0.001
    for i in range(0,nro_exp):        
        tempo_inicial_vi = time()
        V_opt, pi_opt = laoObject.lao_estrela(epsilon, ganma)
        tempo_final_vi = time()    
        tempo_execucao_vi = tempo_final_vi-tempo_inicial_vi
        print('*********************************************')
        print('  Ambiente: ' + enviroment)
        print('  Experimento:'+ str(i+1))
        print('  Ganma: ' + str(ganma))
        print('  Epsilon: '+ str(epsilon))
        print('  Tempo VI: '+ str(tempo_execucao_vi/360))
        print('*********************************************')        
        # mostrar solucao e salvar resultados em arquivo        
        results = laoObject.printSolution(V_opt, pi_opt)
        results.to_csv('results_'+enviroment+'_exp'+str(i+1)+'_ganma'+str(ganma)+'_epsilon'+str(epsilon)+'_tempo'+str(tempo_execucao_vi/360)+'.txt', index=False) 
        ganma = ganma - 0.2
        epsilon = epsilon*0.1            

#------------------------------------------
# Main()
#------------------------------------------

def main():    
    # teste
    loadData(enviroment_ ='Ambiente0', X_ = 2, Y_ = 5, A_ = 4, So_ = 6, G_ = 10)    

    LAO(ganma =  1, epsilon = 0.001)
    #executarExperimentosLAO(mdpObject = lao, enviroment = 'Ambiente1', nro_exp = 3)
    
    #mdp = LAO(So=1, G=125, X=5, Y=25, enviroment='Ambiente1')
    #executarExperimentosVI(mdpObject = mdp, enviroment = 'Ambiente1', nro_exp = 3)
    
    #mdp = LAO(So=1, G=2000, X=20, Y=100, enviroment='Ambiente2')
    #executarExperimentosVI(mdpObject = mdp, enviroment = 'Ambiente2', nro_exp = 3)
    
    #mdp = LAO(So=1, G=12500, X=50, Y=250, enviroment='Ambiente3')
    #executarExperimentosVI(mdpObject = mdp, enviroment = 'Ambiente3', nro_exp = 3)

if __name__ == "__main__":
    main()