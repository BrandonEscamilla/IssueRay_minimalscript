import os
import autograd.numpy as anp
from pymoo.model.problem import Problem
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
import numpy as np
import time
import json
import pyvista as pv
import multiprocessing
from ray.util.multiprocessing import Pool
import math


"""

    Initialize values and class MasconModel
    
"""

model_int = '165_165_7_17_int.npy'
model_ext = '165_165_100_1_ext.npy'

m_int = np.load(model_int, allow_pickle=True) #agregar como parametro antes de la función
m_ext = np.load(model_ext, allow_pickle=True)

G = 6.674 * 10**-11 #[m3/kg*s2]


class MasconModel(Problem, object):

    def __init__(self, n, np_ext = len(m_ext), np_int = len(m_int), m_l=0.07346 * 10**24, m_i=0, m_ext = m_ext, m_int = m_int):

        # store custom variables needed for evaluation
        self.n = n #Numero de mascons
        self.np_ext = np_ext #Number of points external mesh
        self.np_int = np_int-1 #Number of points internal mesh
        self.m_ext = m_ext #External Mesh 
        self.m_int = m_int #Internal Mesh 

        # define lower and upper bounds -  1d array with length equal to number of variable
        xl = 0 * anp.ones(self.n)
        xu = self.np_int * anp.ones(self.n)
        
        super().__init__(n_var=n, n_obj=1, n_constr=0, xl=xl, xu=xu, evaluation_of="auto")

        # store custom variables needed for evaluation
    
        self.m_l = m_l
        self.m_i = m_l / self.n
   
    def _evaluate(self, x, out, *args, **kwargs):

        f = fitness_function_3(x, np_ext = self.np_ext, np_int = self.np_int , m_l=self.m_l, m_i=self.m_i, m_ext = self.m_ext, m_int = self.m_int)
        out["F"] = f


"""

    Create Genetic Algorithm
    
"""

def runGA():

    start = time.time()

    pop_size = 10
    n_mascons = 1

    algorithm = get_algorithm("ga",
                        pop_size=pop_size,
                        sampling=get_sampling("int_random"),
                        crossover=get_crossover("int_sbx", prob=0.9, eta=3.0),
                        mutation=get_mutation("int_pm", eta=3.0),
                        eliminate_duplicates=True
                        )


    problem = MasconModel(n_mascons)  #n  #n_p

    res = minimize(problem,
                algorithm,
                seed=1,
                save_history=True,
                verbose=True
                )

    end = time.time()
    elapsedTime = (end - start)
    
    resultWithCoord = getCoordinatesByIndex(res.X)
    repeated_values = getRepeatedValues(res.X.tolist())

    print("Elapsed (after compilation) = %s" % elapsedTime)
    print("Best solution found: %s" % res.X)
    print("Function value: %s" % res.F)

    #Save GA info to database 
    gaDict = {
        "properties": {
            "fopt": res.F.tolist(),
            "degree": 165,
            "order": 165,
            "gen": res.algorithm.n_gen,
            "population": pop_size,
            "time": elapsedTime,
            "n_mascons": n_mascons,
            "solution": res.X.tolist(),
            "solution_repeated_values": repeated_values,
        },
        "data": resultWithCoord
    }

    saveJson(gaDict, 'n_' + str(n_mascons)) #save information in a json file

def fitness_function_3(x, np_ext, np_int, m_l, m_i, m_ext, m_int):

    m_ext_tp = [i[0] for i in m_ext] #asignar valor de coordenada de malla externa
    
    m_ext_a_sh = [i[1] for i in m_ext]

    p = Pool()

    args = [[i, m_int, m_i, m_ext_tp, m_ext_a_sh] for i in x]
    
    total = p.map(iterateArrays, args)

    p.close()
    
    p.join()
    
    return np.array(total)

def iterateArrays(args): 
        
        i = args[0]
        m_int = args[1]
        m_i = args[2]
        m_ext_tp = args[3]
        m_ext_a_sh = args[4]

        e_i = np.empty(len(m_ext_tp), dtype=np.float64)
        
        
        for j in range(len(m_ext_tp)): #malla exterior 
            
            a_m = np.empty(len(i), dtype=np.float64) #Array of mascons Acceleration # aceleración igual 0 para el primer test point

            r_tp = np.array(m_ext_tp[j]) #asignar valor de coordenada de malla externa
            
            a_sh = m_ext_a_sh[j] #Get acceleration from position vector in external mesh
            
            for index, k in enumerate(i):
                    # [x1,x2,x3]
                               
                    r_i = np.array(m_int[k]) #asignar valor de coordenada de malla interna
                    
                    r = np.subtract(r_tp, r_i)*1000 #vector r - distance between points todo review value
                    
                    r_norm = math.sqrt((r_i[0] - r_tp[0]) ** 2 + (r_i[1] - r_tp[1]) ** 2 + (r_i[2] - r_tp[2]) ** 2)  * 1000            
                    
                    a_mascon_coord = -((G * m_i)/(r_norm)**3) * r #Compute gravity field

                    a_m[index] = np.sqrt(a_mascon_coord[0]**2 + a_mascon_coord[1]**2 + a_mascon_coord[2]**2)
            
            e_i[j] = np.abs(np.sum(a_m) - a_sh)
            
        e_i_mean = np.mean(e_i)
                
        return e_i_mean
        
def saveJson(dictData, name):
    with open('%s.json' % name, 'w',encoding='UTF-8') as json_file:
        json.dump(dictData, json_file)

def getCoordinatesByIndex(index):
    m_int = np.load('165_165_7_17_int.npy', allow_pickle=True)
    data = {}
    
    for i in index: 
        
        x_int_1 = m_int[i][0]
        y_int_1 = m_int[i][1]
        z_int_1 = m_int[i][2] #asignar valor de coordenada de malla externa
        
        data[str(i)] = [x_int_1, y_int_1, z_int_1]
        
    return data

def getRepeatedValues(array):
    repeatedDict = {}

    for item in array:
        try:    
            repeatedDict[item] += 1
        except:
            repeatedDict[item] = 1
    return repeatedDict

runGA()
