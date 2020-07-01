from Environment import Environment
from Agent import Agent
from ExplorationPolicies import EpsilonGreedyPolicy, GreedyPolicy
from LogicModules import QLearningTabModule, QLearningNeuralModule
from Statistics import StatCollector
from ReadSettings import readSettings
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
import time
import os

# Performs the actual work in a generation by testing each individual and
# Calculating the fitness score
#
# Takes in the id of the worker, an already initialized environment
# (containing the solution key and scenarios that will be used)
# a queue which holds each individual to evaluate, and an output queue
# used to pass back the results to the master thread
def evaluateGenesWorker(workerId, env, inQueue, outQueue):

    while (True):
        geneWithScore = None
        try:
            geneWithScore = inQueue.get(timeout = 1)
        except:
            break

        gene = geneWithScore[1]
        lm = GenePool.createLogicModule(gene)

        env.resetSetIndexes()

        agent = Agent(env, lm)
        learningOverall = agent.train(NUM_SIMULATIONS)
        approxAcc = agent.evaluate(NUM_EVALS)
        geneWithScore[0] = approxAcc + learningOverall

        print(f"[OWN_OUT] \t{datetime.now().strftime('%m/%d - %H:%M:%S')} Finished simulation for gene from worker {workerId}")

        outQueue.put(geneWithScore)


if __name__ == '__main__':

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


    #These parameters can all be adjusted via arguments and it is adviced to do so,
    #instead of adjusting the code directly.

    #Number of repetitions of the genetic pool run
    NUM_REPS = 1

    #Number of simulated problems per run
    NUM_SIMULATIONS = 200
    #Number of (independent) genepool runs per program execution
    NUM_GENS = 2

    #Amount of genes in a pool
    GENE_POOL_SIZE = 64

    #How many individuals are selected by direct copying into the next generation,
    #after having a mutation operation applied to it. It should be noted that
    #GENE_POOL_SIZE - COPIED is equal to the amount of individuals that will be
    #calculated by the crossover of 2 parents
    COPIED = 12

    #Rate of mutation when the mutation operator is applied
    MUTATION = 0.05

    #How many of the best performers of one generation get directly copied into the next
    ELITISM = 8

    #This is where all information of all runs/genepools will be stored
    OUTPUT_DIR = "genepool"

    # This is a flag which is used to see if we are running on peregrine. If it is true,
    # The amount of allowed workers is read from an environment variable. If it is false,
    # this has to be specified seperately.
    PEREGRINE = False

    #The number of workers that can be used. ideally should be specified.
    NUM_PROCS = 1

    # Can be used to only consider one type of exploration policy in the search
    LIMIT_EXPL_TO = None


    #Number of times the agent is tested after NUM_SIMULATION encounters, to get
    #approximate accuracy
    NUM_EVALS = 10_000

    #Load in command line arguments
    try:
        options = getopt.getopt(sys.argv[1:], "", ["poolSize=", "copied=", "mutation=",
                                "elitism=", "dir=", "numReps=", "numGens=",
                                "peregrine=", "numWorkers=", "limitExplPolicy=",
                                "settingsFile=", "numSims=", "numEvals="])

        options = options[0]
        numOptions = len(options)

        optionsProvided = [options[i][0] for i in range(numOptions)]
        currentOption = 0

        while (currentOption < numOptions):
            o = options[currentOption][0]
            a = options[currentOption][1]

            if o == "--settingsFile":
                fileSettings = readSettings(str(a))
                for newOption, newArgument in fileSettings:
                    if newOption not in optionsProvided:
                        options.append([newOption, newArgument])

                numOptions = len(options)

            elif o == "--poolSize":
                GENE_POOL_SIZE = int(a)
            elif o == "--copied":
                COPIED = int(a)
            elif o == "--mutation":
                MUTATION = float(a)
            elif o == "--elitism":
                ELITISM = int(a)
            elif o == "--dir":
                OUTPUT_DIR = str(a)
            elif o == "--numReps":
                NUM_REPS = int(a)
            elif o == "--numGens":
                NUM_GENS = int(a)
            elif o == "--numSims":
                NUM_SIMULATIONS = int(a)
            elif o == "--numEvals":
                NUM_EVALS = int(a)
            elif o == "--peregrine":
                option = str(a).lower()
                if (option == "true" or option == "t"):
                    NUM_PROCS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
                    PEREGRINE = True
                elif (option == "false" or option == "f"):
                    PEREGRINE = False
            elif o == "--numWorkers":
                NUM_PROCS = int(a)
            elif o == "--limitExplPolicy":
                option = str(a).lower()

                if (option == "epsilon" or option == "boltzman"):
                    LIMIT_EXPL_TO = option
                else:
                    print(f"{cm.ERROR} Option for limit exploration policy not recognized ({option}){cm.NORMAL}")
                    exit(1)

            currentOption += 1

    except getopt.GetoptError as err:
        print(str(err))
        exit(1)

    if (PEREGRINE):
        GlobalSettings.printMode = GlobalSettings.PRINT_PEREGRINE
    else:
        GlobalSettings.printMode = GlobalSettings.PRINT_GEN

    #Checking some constraints which must hold:
    numCrossover = GENE_POOL_SIZE - COPIED

    if (numCrossover < 0):
        print(f"{cm.ERROR} Genepool of size {GENE_POOL_SIZE} cannot be made by copying {COPIED} individuals.{cm.NORMAL}")
        exit(1)
    elif (numCrossover % 2 != 0):
        print(f"{cm.ERROR} Genepool size minus the amount of copied individuals must be even, but it is not.{cm.NORMAL}")
        exit(1)

    #Provide the user with some feedback about missing settings
    if NUM_PROCS == 1:
        print(f"{cm.WARNING} running program with only 1 worker. The program will be run sequentially.{cm.NORMAL}")
        time.sleep(5)

    # To allow for easier reproducability
    RANDOM_SEED = 1

    statC = StatCollector.getInstance()
    statC.getSettings().disableLogging()

    #Setup code
    np.random.seed(RANDOM_SEED)
    env = Environment()
    pool = GenePool(genePoolSize = GENE_POOL_SIZE, elitism = ELITISM, copied = COPIED,
                    mutateChance = MUTATION, limitExplTo = LIMIT_EXPL_TO, outputDir = OUTPUT_DIR)

    # This function checks to see if it can find some files in OUTPUT_DIR indicating that
    # there is a run which was not completed yet. This is useful if for some reason the first
    # run of the program was terminated unexpectedly, and allows for a restart where
    # the program left of
    hasRestarted = pool.restart()

    explPolicy = None
    lm = None
    agent = None

    #The number of repetitions (runs):
    for i in range(pool.runCtr, NUM_REPS):
        # Normally we want to reset the current generation for each independent run
        # Except if we started loading again from a file, where we might be halfway into
        # A run
        if not hasRestarted:
            pool.startRun()
            pool.currentGen = 0
            pool.initRandom()
            pool.savePool()
        else:
            hasRestarted = False

        #The simulations themselves
        for i in range(pool.currentGen, NUM_GENS):

            if GlobalSettings.printMode == GlobalSettings.PRINT_NORMAL or GlobalSettings.printMode == GlobalSettings.PRINT_GEN:
                print(f"{cm.BACKED_C} {i} out of {NUM_GENS} generations done.{cm.NORMAL}")

            if GlobalSettings.printMode == GlobalSettings.PRINT_PEREGRINE:
                print(f"[OWN_OUT] {datetime.now().strftime('%m/%d - %H:%M:%S')} Performing simulation {i} out of {NUM_GENS}")

            genes = pool.genePool

            # Generate a solution key, and then generate all scenarios for both training and evaluation,
            # as they will be kept consistent across all individuals
            env.createRandomProblem()
            env.initSets(NUM_SIMULATIONS, NUM_EVALS)

            #Divide the genes over the workers

            outQueue = mp.Queue()
            inQueue = mp.Queue()
            processes = []
            #First we generate a queue, which we fill with the genes
            print(f"[OWN_OUT] {datetime.now().strftime('%m/%d - %H:%M:%S')} Setting up gene queue")
            for gene in genes:
                inQueue.put(gene)

            #Then we launch all workers (see the function above)
            print(f"[OWN_OUT] {datetime.now().strftime('%m/%d - %H:%M:%S')} Starting workers")
            for i in range(0, NUM_PROCS):
                p = mp.Process(target = evaluateGenesWorker, args=(i, env, inQueue, outQueue))
                processes.append(p)
                p.start()

            #Here we wait for the termination of all processes
            for p in processes:
                p.join()

            #And then we collect the results, which are genes + fitness scores, here

            print(f"[OWN_OUT] {datetime.now().strftime('%m/%d - %H:%M:%S')} Collecting new genes")
            pool.genePool = []

            for i in range(0, GENE_POOL_SIZE):
                pool.genePool += [outQueue.get()]

            pool.evolve()
            print(f"[OWN_OUT] {datetime.now().strftime('%m/%d - %H:%M:%S')} finished evolving pool")
        pool.endRun()
