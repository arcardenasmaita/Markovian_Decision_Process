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
Epsilon = 0.000001
Ganma = 0.7
Vi_t = 0
Vi_converg = pd.DataFrame(columns=['Iteracoes','Error'])
Vi_calc = 0

#------------------------------------------
# DEFINICAO DO MUNDO
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
    # T_norte =np.array(T_norte)
    # T_sul =np.array(T_sul)
    # T_leste =np.array(T_leste)
    # T_oeste =np.array(T_oeste)
    T_norte.columns=('s_in', 's_out', 'prob')
    T_sul.columns=('s_in', 's_out', 'prob')
    T_leste.columns=('s_in', 's_out', 'prob')
    T_oeste.columns=('s_in', 's_out', 'prob')        
    
    # T_norte['s_in'] = T_norte['s_in'] -1
    # T_norte['s_out'] = T_norte['s_out'] -1
    # T_sul['s_in'] = T_sul['s_in'] -1
    # T_sul['s_out'] = T_sul['s_out'] -1
    # T_leste['s_in'] = T_leste['s_in'] -1
    # T_leste['s_out'] = T_leste['s_out'] -1
    # T_oeste['s_in'] = T_oeste['s_in'] -1
    # T_oeste['s_out'] = T_oeste['s_out'] -1

#------------------------------------------
# METODOS PARA IMPRIMIR E SALVAR SOLUCAO
#------------------------------------------
    
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
    Vm = pd.DataFrame(V)
    Pm = pd.DataFrame(Pi)
    Vm.to_csv(os.path.join("VI_%s_Val_amb_%s_ep_%s_g_%s_calc_%s" % (nr_exp,Enviroment,Epsilon,Ganma,str(Vi_calc))), index=False) 
    Pm.to_csv(os.path.join("VI_%s_Pol_amb_%s_ep_%s_g_%s_calc_%s" % (nr_exp,Enviroment,Epsilon,Ganma,str(Vi_calc))), index=False) 


    Vm = Vm.to_numpy()
    Vm = Vm.reshape(X,Y)
    
    heat_mapV = sb.heatmap(Vm)
    figure1 = heat_mapV.get_figure()
    figure1.show()
    figure1.savefig(os.path.join("VI_%s_img_V_amb_%s_ep_%s_g_%s.png" % (nr_exp,Enviroment,Epsilon,Ganma)))
    
    # plot convergencia
    # x = Vi_converg.loc[:,'it']
    # y = Vi_converg.loc[:,'error']
    Vi_converg.plot(kind='line',x='it',y='error', color='blue')
    plt.show()

    # plt.title("Experimento en ambiente %s, epsilon %s e ganma %s" % (Enviroment,Epsilon,Ganma))
    # plt.ylabel('Error')
    # plt.xlabel('Iteracoes')
    # plt.plot(x,y, lw=2)        
    # plt.savefig(fname = "VI_%s_img_converg_amb_%s_ep_%s_g_%s.png"%(nr_exp,Enviroment,Epsilon,Ganma), bbox_inches='tight')
    print('{:15} {:15} {:15}'.format('s', 'V(s)', 'pi(s)'))
    print(V)
    print(Pi)


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
    Vm = pd.DataFrame(V)
    Pm = pd.DataFrame(Pi)
    Vm.to_csv(os.path.join("VI_%s_Val_amb_%s_ep_%s_g_%s_calc_%s" % (nr_exp,Enviroment,Epsilon,Ganma,str(Vi_calc))), index=False) 
    Pm.to_csv(os.path.join("VI_%s_Pol_amb_%s_ep_%s_g_%s_calc_%s" % (nr_exp,Enviroment,Epsilon,Ganma,str(Vi_calc))), index=False) 


    Vm = Vm.to_numpy()
    Vm = Vm.reshape(X,Y)
    
    heat_mapV = sb.heatmap(Vm)
    figure1 = heat_mapV.get_figure()
    figure1.show()
    figure1.savefig(os.path.join("VI_%s_img_V_amb_%s_ep_%s_g_%s.png" % (nr_exp,Enviroment,Epsilon,Ganma)))
    
    # plot convergencia
    # x = Vi_converg.loc[:,'it']
    # y = Vi_converg.loc[:,'error']
    Vi_converg.plot(kind='line',x='it',y='error', color='blue')
    plt.show()

    # plt.title("Experimento en ambiente %s, epsilon %s e ganma %s" % (Enviroment,Epsilon,Ganma))
    # plt.ylabel('Error')
    # plt.xlabel('Iteracoes')
    # plt.plot(x,y, lw=2)        
    # plt.savefig(fname = "VI_%s_img_converg_amb_%s_ep_%s_g_%s.png"%(nr_exp,Enviroment,Epsilon,Ganma), bbox_inches='tight')
    print('{:15} {:15} {:15}'.format('s', 'V(s)', 'pi(s)'))
    print(V)
    print(Pi)


#------------------------------------------
# CLASSE MDP
#------------------------------------------
class MDP(object):    
    def __init__(self, So, S, G, StatesNameIndex):                
        self.listStates = pd.DataFrame(StatesNameIndex)
        self.G = G
        self.S = S
        self.So = So
        self.i = 0
        self.T_norte = pd.read_csv(Enviroment+'/Action_Norte.txt', header = None, sep = "   ", engine='python')        
        self.T_sul = pd.read_csv(Enviroment+'/Action_Sul.txt', header = None, sep = "   ", engine='python')        
        self.T_leste = pd.read_csv(Enviroment+'/Action_Leste.txt', header = None, sep = "   ", engine='python')        
        self.T_oeste = pd.read_csv(Enviroment+'/Action_Oeste.txt', header = None, sep = "   ",engine='python')        
        self.T_norte.columns=('s_in', 's_out', 'prob')
        self.T_norte['s_in'] = self.T_norte['s_in'] -1
        self.T_norte['s_out'] = self.T_norte['s_out'] -1
        self.T_sul.columns=('s_in', 's_out', 'prob')
        self.T_sul['s_in'] = self.T_sul['s_in'] -1
        self.T_sul['s_out'] =self. T_sul['s_out'] -1
        self.T_leste.columns=('s_in', 's_out', 'prob')
        self.T_leste['s_in'] = self.T_leste['s_in'] -1
        self.T_leste['s_out'] = self.T_leste['s_out'] -1
        self.T_oeste.columns=('s_in', 's_out', 'prob')        
        self.T_oeste['s_in'] = self.T_oeste['s_in'] -1
        self.T_oeste['s_out'] = self.T_oeste['s_out'] -1
        self.T_norte =np.array(self.T_norte)        
        self.T_sul =np.array(self.T_sul)
        self.T_leste =np.array(self.T_leste)
        self.T_oeste =np.array(self.T_oeste)
        #self.C_matrix = pd.read_csv(Enviroment+'/Cost.txt', header = None, engine='python')                
        #self.C_matrix = np.array(self.C_matrix)#*(-1)
        self.T_matrix = np.zeros([self.S, self.S, 4])    
        self.T_matrix = self.createTmatrix()
    def startState(self):        
        return So
    def isEnd(self, state):        
        return state == self.S-1
    def isGoal(self, state):        
        return state == self.G-1
    # def listStateKeyValue(self):        
    #     for i in range (0, len(self.listStates)):
    #         self.listStates.iloc[i][1] = i            
    #     return self.listStates
    def states(self):        
        return range(0,self.S)
    def actions(self):
        return range(0, 4) # 0:acima, 1:abaixo, 2:direita, 3:izquerda
    def R(self, state, accion):               
        #return float(self.C_matrix[state])
        return 0. if self.isGoal(state) else 1.
        return result    
    def createTmatrix(self):
        self.T_matrix = np.zeros([self.S, self.S, 4])         
        for item in self.T_norte:
            s_in = item[0]
            s_out = item[1]
            prob = item[2]
            for i_in, val_in in enumerate(self.listStates.values):
                for i_out, val_out in enumerate(self.listStates.values):
                    if val_in[0] == s_in and val_out[0] == s_out:
                        self.T_matrix[i_in][i_out][0] = prob                        
        for item in self.T_sul:
            s_in = item[0]
            s_out = item[1]
            prob = item[2]
            for i_in, val_in in enumerate(self.listStates.values):
                for i_out, val_out in enumerate(self.listStates.values):
                    if val_in[0] == s_in and val_out[0] == s_out:
                        self.T_matrix[i_in][i_out][1] = prob      
        for item in self.T_leste:
            s_in = item[0]
            s_out = item[1]
            prob = item[2]
            for i_in, val_in in enumerate(self.listStates.values):
                for i_out, val_out in enumerate(self.listStates.values):
                    if val_in[0] == s_in and val_out[0] == s_out:
                        self.T_matrix[i_in][i_out][2] = prob
        for item in self.T_oeste:
            s_in = item[0]
            s_out = item[1]
            prob = item[2]
            for i_in, val_in in enumerate(self.listStates.values):
                for i_out, val_out in enumerate(self.listStates.values):
                    if val_in[0] == s_in and val_out[0] == s_out:
                        self.T_matrix[i_in][i_out][3] = prob
        return self.T_matrix
    
    def T(self, i_in, i_out, action):       
        return self.T_matrix[i_in][i_out][action]
        

#------------------------------------------
# ALGORITMO VALUE ITERATION
#------------------------------------------

def valueIteration(mdp, epsilon, ganma):
    global Vi_t, Vi_calc, Vi_converg
    Vi_t_ini = time()    
    V = np.zeros([mdp.S]) # almacenar o Voptimo[estado]    
    V_old = np.zeros([mdp.S])
    #V[G-1] = 0.
    pi = np.zeros([mdp.S]) # politicas otimas
    res = np.inf
    Vi_it = 0
    Vi_calc = 0
    Q = np.zeros([S,A])
    

    while res > epsilon:
        np.copyto(V_old, V) 
        for state in mdp.states():
            for action in mdp.actions():
                Q[state,action] = mdp.R(state,action)
                for sNext in mdp.states():
                    Q[state,action] = Q[state,action] + ganma*mdp.T(state,sNext,action)*V_old[sNext]
                    Vi_calc = Vi_calc+1
                #print ('*{:1} {:15}'.format(state, Q[state-1,action-1]))
            V[state] = np.max(Q, axis=1)[state] 
            pi[state] = np.argmax(Q, axis=1)[state]            

        # verificar a convergencia
        res = 0
        for s in mdp.states():
            dif = abs(V_old[s]-V[s])
            if dif > res:
                res = dif
                
        Vi_converg.loc[len(Vi_converg)] = [Vi_it,res]
        Vi_it = Vi_it+1
        # Vm = V.reshape(X,Y)
        # heat_map = sb.heatmap(Vm)
        # plt.show()         
        print('res', res)
        print('calc', Vi_calc)
        print('{:15} {:15} {:15} {:15}'.format('s','indexS', 'V(s)', 'pi(s)'))
        for state in mdp.states():
            print ('{:15} {:15} {:15}'.format(state, V[state], pi[state]), sep="  ")
    return V,pi
  


#------------------------------------------
# CLASSE HIPERGRAFO
#------------------------------------------
# class grafo(object):
#     def __init__(self, So, S, G):  
G_So_I = set()           # conjunto de todos os nodos internos do hipergrafo G_So
G_So_F = set()           # conjunto de todos os nodos fronteira do hipergrafo G_So
G_So_v_I = set()         # conjunto de todos os nodos internos do hipergrafo G_So com a função V(s)
G_So_v_F = set()         # conjunto de todos os nodos fronteira do hipergrafo G_So com a função V(s)
V = {}                   # diccionario de valores para cada tupla (s_n, Vs_n)
Lao_t = 0
nodes_G = pd.DataFrame(columns=['s_in','s_out','prob','action','cost'])

def succesorFor(state, action):        
        # retorna os estados sucesores para um determinado nó com uma açao      
        action = int(action)
        result = set()
        #if self.isEnd(state):            
        #    result.append((state, state, 0.))
        if action == 0:
            for item in T_norte:
                if item[0] == state:
                    result.add(item[1])
        if action == 1:
            for item in T_sul:
                if item[0] == state:
                    result.add(item[1])
        if action == 2:
            for item in T_leste:
                if item[0] == state:
                    result.add(item[1])
        if action == 3:
            for item in T_oeste:
                if item[0] == state:
                    result.add(item[1])
        return result
        
def visitNode(node):
    global nodes_G
    # buscar nodos siguientes ao node
    result = pd.DataFrame(columns=['s_in','s_out','prob','action','cost'])
    if not isGoal(node):            
        df = T_norte
        for index, row in df.iterrows():        
            if row[0] == node:
                result.loc[len(result)] = [row[0],row[1],row[2],0,-1.]
        df = T_sul
        for index, row in df.iterrows():        
            if row[0] == node:
                result.loc[len(result)] = [row[0],row[1],row[2],1,-1.]
        df = T_leste
        for index, row in df.iterrows():        
            if row[0] == node:
                result.loc[len(result)] = [row[0],row[1],row[2],2,-1.]
        df = T_oeste
        for index, row in df.iterrows():        
            if row[0] == node:
                result.loc[len(result)] = [row[0],row[1],row[2],3,-1.]
                
        nodes_G = pd.concat([nodes_G, result], axis=0)
    return result    

def isGoal(node):
    return node == G

def nodes(I, F):
    return I.union(F)

def initializeGraph():
    global V, G_So_F, G_So_v_F
    V = {} #pd.DataFrame(columns=['S','V']) 
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

def listAdjacent(state):
    result = visitNode(state)
    return result.iloc[:,[1]]
    #return {7,1}
    # retorna uma lista da tupla (nextState, action)
    


#------------------------------------------
# ALGORITMO LAP*
#------------------------------------------

def LAO(epsilon, ganma):    
    global G_So_I, G_So_F, G_So_v_I, G_So_v_F, V
    # global Epsilon, Ganma
    # Epsilon = epsilon_
    # Ganma = ganma_
    
    # 1. inicializa uma ´arvore com o n´o raiz s0    
    initializeGraph()
    pi_So_v = set()
    Z = set()   
    G_So = G_So_I.union(G_So_F)
    V_So = []
    pi_So = []
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
        # expandir s
        adj = visitNode(s)
        
        #adicionar os nos expandidos a fronteira caso eles sejam alcanzaveis
        for index, row in adj.iterrows():        
            if row[2] > 0 and not(row[0] in G_So_I): 
                #row['s_in','s_out','prob','action','cost'])
                G_So_F.add(row[0])
                V[row[0]] =  row[4] + heuristic(row[1])*row[4] # aplicar heuristica aos nos fronteira
        
        # e) atualiza estados em G_So
        G_So = G_So_I.union(G_So_F)
        G_So_v = G_So_v_I.union(G_So_v_F)
        
        # f) todos os estados que podem alcancar s com melhores acoes
        Z = G_So_v.copy()
        Z.add(s)
        
        df = pd.DataFrame(Z, columns=['state'])
        df['index']=range(0,len(Z))
        
        # g) atualiza V para todo S em Z, aplicar VI      
        mdp = MDP(So = So, S = len(Z), G=G, StatesNameIndex = df)    # MDP(So, S, G, listState):
        V_So, pi_So = valueIteration(mdp, ganma, epsilon)
        
        # quem que está em G_So_v e que consegui chegar a s
        # atualizar V
        for index, val in enumerate(V): 
             V[val] = V_So[df.iloc[index][1]] + V[val]
        
        # h) reconstroi G_So_v com os novos estados alcanzáveis
        for index, val in enumerate(V_So): 
            nextstates = succesorFor(s, pi_So[index])
            G_So_F.union(nextstates)
        
        G_So_v_F.union(succesorFor(s, pi_So[np.argmax(V_So)]))
        pi_So_v.add(pi_So[np.argmax(V_So)])
        nodesForVisit = G_So_F.intersection(G_So_v_F)
    
    # 3. retorne a política parcial desde So    
    return Z
    
#------------------------------------------
# Main()
#------------------------------------------

def main():    
    #loadData(enviroment ='Ambiente0v', x = 2, y = 5, a = 4, so = 6, g = 10)        
    # loadData(enviroment ='Ambiente1', x = 5, y = 25, a = 4, so = 1, g = 101)    
    # mdp = MDP(So, S, G)    
    # V,pi = valueIteration(mdp, ganma = 0.7, epsilon = 0.00001)    
    # printSolution(V,pi,1)
    
    loadData(enviroment ='Ambiente0', x = 2, y = 5, a = 4, so = 6, g =10 )    
    #mdp = MDP(So, S, G)    
    LAO(ganma = 0.7, epsilon = 0.00001)    
    #printSolution(V,pi,1)
    
    
if __name__ == "__main__":
    main()