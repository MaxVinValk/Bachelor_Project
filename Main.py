
from Environment import Environment
from Agent import Agent
from ExplorationPolicies import EpsilonGreedyPolicy
from LogicModules import QLearningTabModule

import numpy as np

# To allow for easier reproducability
np.random.seed(1)

NUM_SIMULATIONS = 100_000_000

env = Environment()

explPolicy = EpsilonGreedyPolicy(epsilon = 1, decayRate = EpsilonGreedyPolicy.getDecay(0.01, NUM_SIMULATIONS), minEpsilon = 0.001)
lm = QLearningTabModule(explorationPolicy = explPolicy, discountFactor = 0.9, learningRate = 0.1)



# An agent is passed the environment and the logic module. The logic module is the brain for the agent,
# which can be swapped out. The environment contains all information with regards to the environment,
# such as which actions are valid and the size of the state-space. The agent reads this information from the environment
# and passes this along to the logic module for setup

agent = Agent(env, lm)
agent.train(NUM_SIMULATIONS)
