#
#   TODO: REFACTORING
#           - Clean up statistics.py
#
#   TODO: FEATURES
#           - Add the ability to load a single class' data
#           - Add the ability to load a single type of data collected
#           - Add class storing settings ?
#


from Environment import Environment
from Agent import Agent
from ExplorationPolicies import EpsilonGreedyPolicy
from LogicModules import QLearningTabModule, QLearningNeuralModule
from Statistics import StatCollector

import numpy as np

# To allow for easier reproducability
np.random.seed(1)

NUM_SIMULATIONS = 1_000_000

env = Environment()

explPolicy = EpsilonGreedyPolicy(epsilon = 1, decayRate = EpsilonGreedyPolicy.getDecay(0.01, NUM_SIMULATIONS), minEpsilon = 0.001)
lm = QLearningNeuralModule(explorationPolicy = explPolicy, discountFactor = 0.9, learningRate = 0.1)
#lm = QLearningTabModule(explorationPolicy = explPolicy, discountFactor = 0, learningRate = 1)


# An agent is passed the environment and the logic module. The logic module is the brain for the agent,
# which can be swapped out. The environment contains all information with regards to the environment,
# such as which actions are valid and the size of the state-space. The agent reads this information from the environment
# and passes this along to the logic module for setup

agent = Agent(env, lm)
agent.train(NUM_SIMULATIONS)


# StatCollector is a singleton class responsible for storing statistics from all sources for a run

#statC = StatCollector.getInstance()

#statC.summarize()
#statC.plotStatistics(averageOver = int(NUM_SIMULATIONS / 100))
