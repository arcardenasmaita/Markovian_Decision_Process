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
# Definição de classes
#------------------------------------------
class world(object):
    S, So, G, X, Y, A, enviroment
    T_norte, T_sul, T_leste, T_oeste, C_matrix
    def __init__(self, enviroment_, X_, Y_, A_, So_, G_):
        global S, So, G, X, Y, A, enviroment
        global T_norte, T_sul, T_leste, T_oeste, C_matrix
        S = X_*Y_   # S = numero de estados
        So = So_ # So = estado inicial
        G = G_   # G = estado objetivo
        X = X_   # X = eixo x
        Y = Y_   # Y = eixo y
        A = A_   # nro de acoes posiveis        
        enviroment = enviroment_
        T_norte = pd.DataFrame() # Probabiliade de transicao para a acao Norte
        T_sul = pd.DataFrame()   # Probabiliade de transicao para a acao Sur
        T_leste = pd.DataFrame() # Probabiliade de transicao para a acao Leste 
        T_oeste = pd.DataFrame() # Probabiliade de transicao para a acao Oeste
        C_matrix = pd.DataFrame()       # matriz de custo        
        self.loadData()   # carregar os dados da probalidade e custo por acao
        
    def actions(self):
        return range(1, self.A) # 1:norte, 2:sul, 3:leste, 4:oeste

    def loadData(self, enviroment, X, Y):   
        global S, x, y
        global T_norte, T_sul, T_leste, T_oeste, C_matrix
        
        T_norte = pd.read_csv(enviroment+'/Action_Norte.txt', header = None, sep = "   ", engine='python')        
        T_sul = pd.read_csv(enviroment+'/Action_Sul.txt', header = None, sep = "   ", engine='python')        
        T_leste = pd.read_csv(enviroment+'/Action_Leste.txt', header = None, sep = "   ", engine='python')        
        T_oeste = pd.read_csv(enviroment+'/Action_Oeste.txt', header = None, sep = "   ",engine='python')        
        C_matrix = pd.read_csv(enviroment+'/Cost.txt', header = None, engine='python')                
        T_norte.columns=('s_in', 's_out', 'prob')
        T_sul.columns=('s_in', 's_out', 'prob')
        T_leste.columns=('s_in', 's_out', 'prob')
        T_oeste.columns=('s_in', 's_out', 'prob')
        S = X*Y
        x = X
        y = Y
        
    def T(s_in, s_out, action):                
        if action == 0:
            return float(T_norte[(T_norte['s_in'] == s_in) & (T_norte['s_out'] == s_out)]["prob"]) if not T_norte[(T_norte['s_in'] == s_in) & (T_norte['s_out'] == s_out)].empty else 0
        elif action == 1:
            return float(T_sul[(T_sul['s_in'] == s_in) & (T_sul['s_out'] == s_out)]["prob"]) if not T_sul[(T_sul['s_in'] == s_in) & (T_sul['s_out'] == s_out)].empty else 0    
        elif action == 2:
            return float(T_leste[(T_leste['s_in'] == s_in) & (T_leste['s_out'] == s_out)]["prob"]) if not T_leste[(T_leste['s_in'] == s_in) & (T_leste['s_out'] == s_out)].empty else 0
        elif action == 3:
            return float(T_oeste[(T_oeste['s_in'] == s_in) & (T_oeste['s_out'] == s_out)]["prob"]) if not T_oeste[(T_oeste['s_in'] == s_in) & (T_oeste['s_out'] == s_out)].empty else 0
        else:
            return 0
        
        
class Hipergraph(object): # representacao de um hipergrafo de conetividade G_S_o
    G, So, I, F, mundo
    def __init__(self, mundo_):
        global G, So, I, F, mundo    
        G = mundo.G
        So = mundo.So
        I = set() # nós internos do grafo / nós expandidos
        F = set([So]) # nós na frontera do grafo
        mundo = mundo_
        
    def expand(self, node):
        # buscar nodos siguientes ao node
        
        return ''
    def isGoal(self, node):
        return node == self.G
    def nodes(self):
        return I.union(F)
        
        
class mdp_LAO(object):
    V, So, G, G_So, G_So_v, Z, mundo
    def __init__(self, mundo_):  
        global V, So, G, G_So, G_So_v, Z, mundo
        mundo = mundo_ # ambiente do agente
        V = {}      #diccionario de valores para cada par [s_n, Vs_n]
        So = mundo_.So   # So = estado inicial
        G = mundo.G_      # G = estado objetivo         
        G_So = Hipergraph(So, G) # I U F
        G_So_v = Hipergraph(So, G)        
        Z = set()   
        
        #A_Tree = pd.DataFrame()#

    #def states(self):
    #    return range(1, self.S+1)
    def C(self, state, action):              
        return float(C_matrix[0][state])
    
    def heuristic(self, state): # H(s)
        return 0
    def accumulated_C(self): # C(s)
        return 0
    def function_V(self): # V(s)
        #H(s) ≤ V (s)
        return 0
    
    ### Algoritmos
            
    def lao_estrela(self, epsilon, ganma):
        global V, So, G, G_So, G_So_v, Z
        #1. inicializa uma ´arvore com o n´o raiz s0
        V = {So:0} 
        V[So] = self.heuristic(So) 
        G_So = Hipergraph(So, G)
        G_So_v = Hipergraph(So, G)        
                
        Z = set()   
        A_Tree = pd.DataFrame()#
                
        #2. repete enquanto existam nos na fronteira e G_So_v
        nodesForVisit = G_So.F.intersection(G_So_v.nodes)
        while s in nodesForVisit and s != self.G:
            #a) escolha um n´o folha para expandir
            s = nodeForVisit.pop()
                        
            
            #b) expanda o n´o folha escolhido
            
            
            #1. aumenta a fronteira de G^s0,
            #2. atualiza V , e
            #3. reconstr´oi G^Vs0.
            #c) fa¸ca anota¸c~oes necess´arias        
        
        return ""


#------------------------------------------
# Métodos inicio
#------------------------------------------

# declarr variaveis globais

A = 4   # nro de acoes posiveis        
S = 0   # nro total de estados
So = 0  # estado inicial
G = 0   # estado objetivo
x = 0   # número de colunas da matriz
y = 0   # número de filas da matriz
mundo = World()


def printSolution(V, pi):               
    return "results"
    

### Experimentos
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


def main():    
    # teste
    mundo = World.loadData(enviroment='Ambiente0', X=2, Y=5, A=4, So = 6, G = 10)    
    lao = mdp_LAO(mundo)
    
    
    
    #executarExperimentosLAO(mdpObject = lao, enviroment = 'Ambiente1', nro_exp = 3)
    
    #mdp = LAO(So=1, G=125, X=5, Y=25, enviroment='Ambiente1')
    #executarExperimentosVI(mdpObject = mdp, enviroment = 'Ambiente1', nro_exp = 3)
    
    #mdp = LAO(So=1, G=2000, X=20, Y=100, enviroment='Ambiente2')
    #executarExperimentosVI(mdpObject = mdp, enviroment = 'Ambiente2', nro_exp = 3)
    
    #mdp = LAO(So=1, G=12500, X=50, Y=250, enviroment='Ambiente3')
    #executarExperimentosVI(mdpObject = mdp, enviroment = 'Ambiente3', nro_exp = 3)

if __name__ == "__main__":
    main()