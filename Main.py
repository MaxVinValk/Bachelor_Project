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

import numpy as np
import getopt
import sys

#Number of simulated problems per run
NUM_SIMULATIONS = 200
#Number of (independent) runs per program execution
NUM_REPETITIONS = 250

# To allow for easier reproducability
RANDOM_SEED = 1


#Allow the user to change these variables instead of using their defaults

statC = StatCollector.getInstance()

try:
    options = getopt.getopt(sys.argv[1:], "", ["numSimulations=", "numRepetitions=", "randomSeed=", "dataRoot="])
    for o, a in options[0]:
        print(o)
        if o == "--numSimulations":
            NUM_SIMULATIONS = int(a)
        elif o == "--numRepetitions":
            NUM_REPETITIONS = int(a)
        elif o == "--randomSeed":
            RANDOM_SEED = int(a)
        elif o == "--dataRoot":
            statC.setDataRoot(str(a))


except getopt.GetoptError as err:
    print(str(err))
    exit(1)

print(f"{cm.WARNING} Have you re-enabled data collection?")


#Setup code

np.random.seed(RANDOM_SEED)
env = Environment()

explPolicy = None
lm = None
agent = None

EPSILON_DECAY = 0.4519835#EpsilonGreedyPolicy.getDecay(targetEpsilon = 0.01, numEpisodes = 25)
MIN_EPSILON = 0.01
START_EPSILON = 0.5159773
DISCOUNT_FACTOR = 0
LEARNING_RATE = 0.14304498

MIN_REPLAY_MEMORY_SIZE = 97
MINIBATCH_SIZE = 22
LAYERS = 2
NODES_IN_LAYER = 14

statC.startSession()
statC.addSessionData("random_seed", RANDOM_SEED)
statC.addSessionData("sims_per_run", NUM_SIMULATIONS)
statC.addSessionData("start_epsilon", START_EPSILON)
statC.addSessionData("epsilon_decay", EPSILON_DECAY)
statC.addSessionData("discount_factor", DISCOUNT_FACTOR)
statC.addSessionData("learning_rate", LEARNING_RATE)
statC.addSessionData("Model info", "2 x 32, min replay: 32, batch size: 16")

#The simulations themselves
for i in range(0, NUM_REPETITIONS):

    print(f"{cm.BACKED_C} {i} out of {NUM_REPETITIONS} simulations done.{cm.NORMAL}")


    statC.startRun()
    env.createRandomProblem()

    explPolicy = EpsilonGreedyPolicy(epsilon = START_EPSILON, decayRate = EPSILON_DECAY, minEpsilon = MIN_EPSILON)
    lm = QLearningNeuralModule( explorationPolicy = explPolicy, discountFactor = DISCOUNT_FACTOR, learningRate = LEARNING_RATE,
                                minReplayMemorySize = MIN_REPLAY_MEMORY_SIZE, miniBatchSize = MINIBATCH_SIZE,
                                layers = LAYERS, nodesInLayer = NODES_IN_LAYER)

    #lm = QLearningNeuralModule(explorationPolicy = explPolicy, discountFactor = DISCOUNT_FACTOR, learningRate = LEARNING_RATE)
    agent = Agent(env, lm)
    agent.train(NUM_SIMULATIONS)
