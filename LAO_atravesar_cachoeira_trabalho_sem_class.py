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
V = pd.DataFrame()                # diccionario de valores para cada tupla (s_n, Vs_n)
epsilon = 0.001
ganma = 1
T = {}

# enviroment ='Ambiente1'
# X = 5
# Y = 25
# A = 4
# S=125
# So = 1
# G = 125
#------------------------------------------
# Métodos para definir o mundo
#------------------------------------------

def loadData(enviroment_, X_, Y_, A_, So_, G_):   
    global A, G, So, S, x, y, enviroment
    global T_norte, T_sul, T_leste, T_oeste, C_matrix, T
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
    # T_norte =np.array(T_norte)
    # T_sul =np.array(T_sul)
    # T_leste =np.array(T_leste)
    # T_oeste =np.array(T_oeste)
    
    # T=pd.DataFrame(columns = ['listT'], index = range(1,S+1) )
    # T['listT'] = {}

    # for item in T_norte:
    #     tripla = (item[1],item[2],1.)
    #     if pd.isnan(T.iloc[int(item[0])]['listT']):
    #         tripla = (item[1],item[2],1.)
    #         T.iloc[int(item[0])] =T.iloc[int(item[0])]['listT'].append(tripla)
        
    
    
def printSolution(self, Rm,X,Y):    
    Rm = Rm.drop([3], axis=1) 
    Rm = Rm.to_numpy()
    
    Vm = Rm[:,1]
    Pm = Rm[:,2]
    
    Vm = Vm.reshape(X,Y)
    #pi_grafico = Pm.reshape(Y,X)
    heat_map = sb.heatmap(Vm)
    figure = heat_map.get_figure()
    figure.show()
    figure.savefig(os.path.join("Img_ambiente_%s_ep_%s_g_%s.png" % (enviroment,epsilon,ganma)))
    
def actions(A):
    return range(1, A+1) # 1:norte, 2:sul, 3:leste, 4:oeste

def T(s_in, s_out, action):                
    if action == 1:
        df =  T_norte[(T_norte['s_out'] == s_out) & (T_norte['s_in'] == s_in) ]
        return float(df["prob"]) if not df.empty else 0
    elif action == 2:
        df = T_sul[(T_sul['s_out'] == s_out) & (T_sul['s_in'] == s_in)]
        return float(df["prob"]) if not df.empty else 0    
    elif action == 3:
        df=T_leste[(T_leste['s_out'] == s_out) & (T_leste['s_in'] == s_in) ]
        return float(df["prob"]) if not df.empty else 0
    elif action == 4:
        df=T_oeste[(T_oeste['s_out'] == s_out) & (T_oeste['s_in'] == s_in) ]
        return float(df["prob"]) if not df.empty else 0
    else:
        return 0
    # return 2
def states(S):
    return range(1, S+1)

def C(state):              
    if isGoal(state):#-1
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
    global V, G_So_F, G_So_v_F
    V = pd.DataFrame(columns=['S','V']) 
    V.loc[len(V)] = [So, heuristic(So)]
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

def valueIteration(S_list, G):
    V = pd.DataFrame(0, index = range(0,len(S_list)), columns = ['S', 'V']) # almacenar o Voptimo[estado]
    V.loc[:,'S'] = S_list      
    V_old = pd.DataFrame(0, index = range(0,len(S_list)), columns = ['S','V'])
    V_old.loc[:,'S'] = S_list   
    pi = pd.DataFrame(0, index=range(0,len(S_list)), columns=['S','pi']) # politicas otimas
    pi.loc[:,'S'] = S_list   
    
    res = np.inf
    Q = pd.DataFrame(0, index=range(0,len(S_list)), columns = actions(A))
    k = 0
    calc = 0
    # criar matriz temporal para S_list
    T_z = np.zeros(shape=(len(S_list),len(S_list),4))
    for act in actions(A):
        for indexX, itemX in enumerate(S_list):
            for indexY, itemY in enumerate(S_list):
                T_z[indexX,indexY,act-1] = T(itemX, itemY,act)
    
    
    while res > epsilon: #or calc < 1000:
        k=k+1
        V_old = V.copy()
        for index_s, item_s in enumerate(S_list):
            # if item_s == G:
            #     V.loc[index_s,'V'] = 0.
            # else:
            for action in actions(A):
                Q.loc[index_s,action] = C(item_s)
                for index_next in range(0,len(S_list)):
                    #print(index_s,action)
                    Q.loc[index_s,action] = Q.loc[index_s,action] + ganma*T_z[index_s, index_next,action-1]*V_old.loc[index_next,'V']
                    calc = calc+1
            V.loc[index_s,'V'] = np.max(Q, axis=1)[index_s] 
            pi.loc[index_s,'pi'] = Q.idxmax(axis=1)[index_s]    
            print('V')
            print(V)
            print('V_old')
            print(V_old)
            print('pi')
            print(pi)
            #mostrar grafico da evolucao
            # Vm = np.array(Q)
            # heat_map = sb.heatmap(Vm)
            # figure = heat_map.get_figure()    
            # figure.savefig(os.path.join("Img_ambiente_%s_ep_%s_g_%s.png" % (enviroment,epsilon,ganma)))
       
        # verificar a convergencia
        res = 0
        for s in range(0, len(S_list)):
            dif = abs(V_old.loc[s,'V']-V.loc[s,'V'])
            if dif > res:
                res = dif
        print(calc," ", dif)
    return V,pi

#------------------------------------------
# Algoritmo LAO*
#------------------------------------------

def LAO(epsilon_, ganma_):    
    global G_So_I, G_So_F, G_So_v_I, G_So_v_F, V
    global epsilon, ganma
    epsilon = epsilon_
    ganma = ganma_
    
    # 1. inicializa uma ´arvore com o n´o raiz s0    
    initializeGraph()
    pi_So_v = set()
    Z = set()   
    G_So = G_So_I.union(G_So_F)
                
    # 2. repete enquanto existam nos na fronteira e G_So_v
    nodesForVisit = G_So_F.intersection(G_So_v_F)
    s = 0
    while len(nodesForVisit) > 0 and not isGoal(s):
        # a) escolha um no folha para expandir
        s = nodesForVisit.pop()
        # b) retira s do conjunto de nós fronteira
        G_So_F.remove(s)
        # d) adiciona s ao conjunto de nós internos
        G_So_I.add(s)
        # c) expandir, adicionar a F os adjacentes com T>0
        adj = listAdjacent(s)
        
        for node_adj in adj:
            G_So_F.add(node_adj) #adicionar os nos expandidos a fronteira
            V.loc[len(V)] = [node_adj, heuristic(node_adj)] # aplicar heuristica aos nos fronteira
        
        # e) atualiza estados em G_So
        G_So = G_So_I.union(G_So_F)
        
        # f) todos os estados que podem alcancar s, melhores acoes
        Z = G_So.copy()
        Z.add(s)
        
        # g) atualiza V para todo S em Z, aplicar VI       
        V_So, pi_So = valueIteration(Z,s)
        # quem que está em G_So_v e que consegui chegar a s
        #V.append([V_So])
        # h) reconstroi G_Vs0
        
        nodesForVisit = G_So_F.intersection(G_So_v_F)
    
    # 3. retorne a política parcial desde So    
    return pi_So_v

#------------------------------------------
# Métodos para executar os experimentos
#------------------------------------------

def executarExperimentosLAO(laoObject,nro_exp):
    global epsilon, ganma
    fig, axs = plt.subplots(nro_exp, 1, sharex=True)
    
    for i in range(1,nro_exp+1):
        tempo_inicial_vi = time()
        V_opt, pi_opt, converg = valueIteration(epsilon, ganma)
        tempo_final_vi = time()    
        tempo_execucao_vi = tempo_final_vi-tempo_inicial_vi
        print('*********************************************')
        print('  Ambiente: ' + enviroment)
        print('  Experimento: ' + str(nro_exp))
        print('  Ganma: ' + str(ganma))
        print('  Epsilon: '+ str(epsilon))
        print('  Tempo VI: '+ str(tempo_execucao_vi/360))
        print('  Nro iteracoes: ' + str(np.max(converg.loc[:,'Iteracoes'])))
        print('  Erro converg: ' + str(np.min(converg.loc[:,'Erro'])))
        print('  Nro calculos de Q: ' + str(np.max(converg.loc[:,'q'])))
        print('  Nos expandidos: ' + str(1))
        print('*********************************************')        
        # mostrar solucao e salvar resultados em arquivo        
        printSolution(converg)
        # converg = pd.DataFrame(columns=['Iteracoes', 'Erro','q'])
        # converg.loc[9] = [2,2,3]
        # converg.loc[10] = [3,4,4]
        # converg.loc[12] = [4,6,4]
        
        # plot figuras
        axs[i-1].plot(converg.loc[:,'Iteracoes'], converg.loc[:,'Erro'], lw=2)        
        axs[i-1].set_title('Experimento'+str(i))
        
        # salvar resultados
        converg.to_csv('Amb_'+enviroment+'_exp_'+str(i)+'_g_'+str(ganma)+'_e_'+str(epsilon)+'_t'+str(tempo_execucao_vi/360)+'_it_'+str(np.max(converg.loc[:,'Iteracoes']))+'_calq_'+str(np.max(converg.loc[:,'q']))+'.txt', index=False) 
        
        # variar os parámetros
        ganma = ganma - 0.2
        epsilon = epsilon*0.01      
    
    for ax in axs.flat:
        ax.set(xlabel='Iteracoes', ylabel='Erro')
        ax.label_outer()
        
    plt.savefig(os.path.join("ambiente_%s_ep_%s_g_%s.png" % (enviroment,epsilon,ganma)), bbox_inches='tight')

#------------------------------------------
# Main()
#------------------------------------------

def main():    
    # teste
    loadData(enviroment_ ='Ambiente1', X_ = 2, Y_ = 5, A_ = 4, So_ = 6, G_ = 10)    

    LAO(ganma_ =  1, epsilon_ = 0.001)
    #executarExperimentosLAO(mdpObject = lao, enviroment = 'Ambiente1', nro_exp = 3)
    
    #mdp = LAO(So=1, G=125, X=5, Y=25, enviroment='Ambiente1')
    #executarExperimentosVI(mdpObject = mdp, enviroment = 'Ambiente1', nro_exp = 3)
    
    #mdp = LAO(So=1, G=2000, X=20, Y=100, enviroment='Ambiente2')
    #executarExperimentosVI(mdpObject = mdp, enviroment = 'Ambiente2', nro_exp = 3)
    
    #mdp = LAO(So=1, G=12500, X=50, Y=250, enviroment='Ambiente3')
    #executarExperimentosVI(mdpObject = mdp, enviroment = 'Ambiente3', nro_exp = 3)

if __name__ == "__main__":
    main()