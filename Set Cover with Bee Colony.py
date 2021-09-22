import sys
import copy
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import random

All_A = []
def setCover(vector):

    A = ['a','b','c','d','e','f','g']
    A_1 = ['a','b','f','g']
    A_2 = ['a','b','g']
    A_3 = ['a','b','c']
    A_4 = ['e','f','g']
    A_5 = ['f','g']
    A_6 = ['d','f']
    A_7 = ['d']
    global All_A
    All_A = [A_1, A_2, A_3, A_4, A_5, A_6, A_7]
    
    A_1_Cost = 1
    A_2_Cost = 1
    A_3_Cost = 1
    A_4_Cost = 1
    A_5_Cost = 1
    A_6_Cost = 1
    A_7_Cost = 1
    
    '''
    A_1_Cost = 2
    A_2_Cost = 2
    A_3_Cost = 3
    A_4_Cost = 3
    A_5_Cost = 2
    A_6_Cost = 3
    A_7_Cost = 1
    '''
    All_A_Cost = [A_1_Cost, A_2_Cost, A_3_Cost, A_4_Cost, A_5_Cost, A_6_Cost, A_7_Cost]
    
    vector = np.array(vector)
    round_vector = [round(num) for num in vector]
 
    selected_elements = []
    fitness = 0
    for i in range(len(vector)):
        if(round_vector[i] == 1):
            selected_elements = list(set().union(selected_elements, All_A[i]))
            fitness = fitness + All_A_Cost[i]
       
    selected_elements.sort()
    if selected_elements == A:
        fitness = fitness # redundant code
    else:
        fitness = sum(All_A_Cost)
        
    return fitness

def printBestResult(resultVector):
    print("Best Result by far:")
    round_vector = [round(num) for num in resultVector]
    for i in range(len(round_vector)):
        if(round_vector[i] == 1):
            print(All_A[i])

    print("-------")
    
    
class Bee(object):
    def __init__(self, lower, upper, fun):
        self._random(lower, upper)
        self.value = fun(self.vector)
        self._fitness()
        self.counter = 0

    def _random(self, lower, upper):
        self.vector = []
        for i in range(len(lower)):
            self.vector.append( lower[i] + random.random() * (upper[i] - lower[i]) )

    def _fitness(self):
        if (self.value >= 0):
            self.fitness = 1 / (1 + self.value)
        else:
            self.fitness = 1 + abs(self.value)

class BeeModel(object):
    def run(self):
        cost = {}; cost["best"] = []; cost["mean"] = []
        for itr in range(self.max_itrs):
            for index in range(self.size):
                self.send_employee(index)

            self.send_onlookers()
            self.send_scout()
            self.find_best()
            cost["best"].append( self.best )
            cost["mean"].append( sum( [ bee.value for bee in self.population ] ) / self.size )

        return cost

    def __init__(self,lower, upper,fun= None, numb_bees = 15, max_itrs = 50 ,max_trials = None, selfun = None):
        self.size = int((numb_bees + numb_bees % 2))
        self.dim = len(lower)
        self.max_itrs = max_itrs
        if (max_trials == None):
            self.max_trials = 0.6 * self.size * self.dim
        else:
            self.max_trials = max_trials
        self.selfun = selfun
        self.evaluate = fun
        self.lower    = lower
        self.upper    = upper
        self.best = sys.float_info.max
        self.solution = None
        self.population = [ Bee(lower, upper, fun) for i in range(self.size) ]
        self.find_best()
        self.compute_probability()

    def find_best(self):
        values = [ bee.value for bee in self.population ]
        index  = values.index(min(values))
        if (values[index] < self.best):
            self.best     = values[index]
            self.solution = self.population[index].vector
            printBestResult(self.solution )

    def compute_probability(self):
        values = [bee.fitness for bee in self.population]
        max_values = max(values)
        if (self.selfun == None):
            self.probas = [0.9 * v / max_values + 0.1 for v in values]
        else:
            self.probas = self.selfun(values)

        return [sum(self.probas[:i+1]) for i in range(self.size)]

    def send_employee(self, index):
        zombee = copy.deepcopy(self.population[index])
        d = random.randint(0, self.dim-1)
        bee_ix = index;
        while (bee_ix == index): bee_ix = random.randint(0, self.size-1)
        zombee.vector[d] = self._mutate(d, index, bee_ix)
        zombee.vector = self._check(zombee.vector, dim=d)
        zombee.value = self.evaluate(zombee.vector)
        zombee._fitness()
        if (zombee.fitness > self.population[index].fitness):
            self.population[index] = copy.deepcopy(zombee)
            self.population[index].counter = 0
        else:
            self.population[index].counter += 1

    def send_onlookers(self):
        numb_onlookers = 0; beta = 0
        while (numb_onlookers < self.size):
            phi = random.random()
            beta += phi * max(self.probas)
            beta %= max(self.probas)
            index = self.select(beta)
            self.send_employee(index)
            numb_onlookers += 1

    def select(self, beta):
        probas = self.compute_probability()
        for index in range(self.size):
            if (beta < probas[index]):
                return index

    def send_scout(self):
        trials = [ self.population[i].counter for i in range(self.size) ]
        index = trials.index(max(trials))
        if (trials[index] > self.max_trials):
            self.population[index] = Bee(self.lower, self.upper, self.evaluate)
            self.send_employee(index)

    def _mutate(self, dim, current_bee, other_bee):
        return self.population[current_bee].vector[dim] +  (random.random() - 0.5) * 2 * (self.population[current_bee].vector[dim] - self.population[other_bee].vector[dim])

    def _check(self, vector, dim=None):
        if (dim == None):
            range_ = range(self.dim)
        else:
            range_ = [dim]
        for i in range_:
            if  (vector[i] < self.lower[i]):
                vector[i] = self.lower[i]
            elif (vector[i] > self.upper[i]):
                vector[i] = self.upper[i]
        return vector

def DrawPlot(cost):
    labels = ["Best Fitness plot", "Mean Fitness Function"]
    plt.figure(figsize=(10, 4));
    plt.plot(range(len(cost["best"])), cost["best"], label=labels[0]);
    plt.scatter(range(len(cost["mean"])), cost["mean"], color='red', label=labels[1]);
    plt.xlabel("Iteration number");
    plt.ylabel("Values");
    plt.legend(loc="best");
    plt.xlim([0,len(cost["mean"])]);
    plt.grid();
    plt.show();

def run():
    ndim = int(7)
    model = BeeModel(lower= [0] *ndim, upper = [1]*ndim, fun  = setCover, numb_bees =  30, max_itrs  =  100 ,)

    cost = model.run()
    DrawPlot(cost)
    print("Fitness: {0}".format(model.best))

if __name__ == "__main__":
    run()
