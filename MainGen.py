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

import numpy as np
import getopt
import sys

if __name__ == '__main__':


    #Number of simulated problems per run
    NUM_SIMULATIONS = 200
    #Number of (independent) genepool runs per program execution
    NUM_GENS = 1
    GENE_POOL_SIZE = 64

    # To allow for easier reproducability
    RANDOM_SEED = 1


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
            print(f"{datetime.now().strftime('%m/%d - %H:%M:%S')} Performing simulation {i} out of {NUM_GENS}")

        genes = pool.genePool
        env.createRandomProblem()

        for i in range(0, len(genes)):

            gene = genes[i][1]
            lm = GenePool.createLogicModule(gene)
            agent = Agent(env, lm)


            #we use the area under the curve as a heuristic for learning speed.
            learningOverall = agent.train(NUM_SIMULATIONS)

            #runAcc = statC.getStatisticLatest("Agent", "guessesAccuracyOverTime")
            #learningOverall = sum(runAcc["data"]) / len(runAcc["data"])

            #We run a number of random simulations with the algorithm and see if it
            #finds the correct answer.
            approxAcc = agent.evaluate(10000)

            pool.genePool[i][0] = approxAcc + learningOverall

            if GlobalSettings.printMode == GlobalSettings.PRINT_MODES[0]:
                print(f"score: {pool.genePool[i][0]}")

            if GlobalSettings.printMode == GlobalSettings.PRINT_MODES[1]:
                print(f"\t{datetime.now().strftime('%m/%d - %H:%M:%S')} Finished simulation for gene {i}")

        pool.evolve()
        print(f"{datetime.now().strftime('%m/%d - %H:%M:%S')} finished evolving pool")
