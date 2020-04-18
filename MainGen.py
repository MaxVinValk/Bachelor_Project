#
#   Statistics:
#       TODO:   Move features from StatCollector w.r.t. plotting and such,
#               to its own class
#
#   allow getDecay to start at a non-1 value
#
#

from Environment import Environment
from Agent import Agent
from ExplorationPolicies import EpsilonGreedyPolicy, GreedyPolicy
from LogicModules import QLearningTabModule, QLearningNeuralModule
from Statistics import StatCollector
from ConsoleMessages import ConsoleMessages as cm
from GenAlg import GenePool
from RunSettings import GlobalSettings
from datetime import datetime

import multiprocessing as mp
#To have the warning output up front
import tensorflow as tf

import numpy as np
import getopt
import sys
import os

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Number of simulated problems per run
NUM_SIMULATIONS = 200
#Number of (independent) genepool runs per program execution
NUM_GENS = 1
GENE_POOL_SIZE = 64

#Processes to be spawned. GENE_POOL_SIZE needs to be divisible by this if
# evaluateGenes is used.
# For evaluateGenesWorker, no such restriction applies.
#
NUM_PROCS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])

#Number of times the agent is tested after NUM_SIMULATION encounters, to get
#approximate accuracy
NUM_EVALS = 10_000

# To allow for easier reproducability
RANDOM_SEED = 1

def evaluateGenesWorker(workerId, env, inQueue, outQueue):

    while (True):
        geneWithScore = None
        try:
            geneWithScore = inQueue.get(timeout = 1)
        except:
            break

        gene = geneWithScore[1]
        lm = GenePool.createLogicModule(gene)
        agent = Agent(env, lm)

        learningOverall = agent.train(NUM_SIMULATIONS)
        approxAcc = agent.evaluate(NUM_EVALS)
        geneWithScore[0] = approxAcc + learningOverall

        if GlobalSettings.printMode == GlobalSettings.PRINT_MODES[0]:
            print(f"score: {geneWithScore[0]}")

        if GlobalSettings.printMode == GlobalSettings.PRINT_MODES[1]:
            print(f"[OWN_OUT] \t{datetime.now().strftime('%m/%d - %H:%M:%S')} Finished simulation for gene from worker {workerId}")

        outQueue.put(geneWithScore)






def evaluateGenes(genes, lowIdx, highIdx, env, outQueue):
    for i in range(lowIdx, highIdx):
        gene = genes[i][1]
        lm = GenePool.createLogicModule(gene)
        agent = Agent(env, lm)

        learningOverall = agent.train(NUM_SIMULATIONS)
        approxAcc = agent.evaluate(NUM_EVALS)
        genes[i][0] = approxAcc + learningOverall

        if GlobalSettings.printMode == GlobalSettings.PRINT_MODES[0]:
            print(f"score: {pool.genePool[i][0]}")

        if GlobalSettings.printMode == GlobalSettings.PRINT_MODES[1]:
            print(f"[OWN_OUT] \t{datetime.now().strftime('%m/%d - %H:%M:%S')} Finished simulation for gene {i}")

    outQueue.put(genes[lowIdx:highIdx])



if __name__ == '__main__':


    #Allow the user to change these variables instead of using their defaults

    statC = StatCollector.getInstance()
    statC.getSettings().disableDataOutput()
    statC.getSettings().disableLogging()

    #Setup code

    np.random.seed(RANDOM_SEED)
    env = Environment()
    pool = GenePool(genePoolSize = GENE_POOL_SIZE)

    if (pool.restart() == False):
        pool.savePool()

    explPolicy = None
    lm = None
    agent = None


    #The simulations themselves
    for i in range(pool.currentGen, NUM_GENS):

        #split

        if GlobalSettings.printMode == GlobalSettings.PRINT_MODES[0]:
            print(f"{cm.BACKED_C} {i} out of {NUM_GENS} simulations done.{cm.NORMAL}")

        if GlobalSettings.printMode == GlobalSettings.PRINT_MODES[1]:
            print(f"[OWN_OUT] {datetime.now().strftime('%m/%d - %H:%M:%S')} Performing simulation {i} out of {NUM_GENS}")

        genes = pool.genePool
        env.createRandomProblem()

        #Divide the genes to processes

        #New method: Using a problem queue
        outQueue = mp.Queue()
        inQueue = mp.Queue()
        processes = []

        print(f"[OWN_OUT] {datetime.now().strftime('%m/%d - %H:%M:%S')} Setting up gene queue")
        for gene in genes:
            inQueue.put(gene)

        print(f"[OWN_OUT] {datetime.now().strftime('%m/%d - %H:%M:%S')} Starting workers")
        for i in range(0, NUM_PROCS):
            p = mp.Process(target = evaluateGenesWorker, args=(i, env, inQueue, outQueue))
            processes.append(p)
            p.start()


        for p in processes:
            p.join()

        print(f"[OWN_OUT] {datetime.now().strftime('%m/%d - %H:%M:%S')} Collecting new genes")
        pool.genePool = []

        for i in range(0, GENE_POOL_SIZE):
            pool.genePool += [outQueue.get()]

        #Old method: equal division over workers
        '''
        genesInPool = int(GENE_POOL_SIZE / NUM_PROCS)
        processes = []
        outQueue = mp.Queue()

        for i in range(0, NUM_PROCS):
            p = mp.Process(target = evaluateGenes, args=(genes, i*genesInPool, (i+1) * genesInPool, env, outQueue))
            processes.append(p)
            p.start()

        pool.genePool = []

        for p in processes:
            p.join()
            pool.genePool += outQueue.get()
        '''
        pool.evolve()
        print(f"[OWN_OUT] {datetime.now().strftime('%m/%d - %H:%M:%S')} finished evolving pool")
