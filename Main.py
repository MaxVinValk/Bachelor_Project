from Environment import Environment
from Agent import Agent
from ExplorationPolicies import EpsilonGreedyPolicy, GreedyPolicy, BoltzmanExplorationPolicy
from LogicModules import QLearningTabModule, QLearningNeuralModule
from Statistics import StatCollector
from ReadSettings import readSettings
from ConsoleMessages import ConsoleMessages as cm
from RunSettings import GlobalSettings

import numpy as np
import getopt
import sys

#Number of simulated problems per run
NUM_SIMULATIONS = 200
#Number of (independent) runs per program execution
NUM_REPETITIONS = 250

# To allow for easier reproducability
RANDOM_SEED = 1

EXPLORATION_POLICY = "epsilon"
LOGIC_MODULE = "nn"

#Allow the user to change these variables instead of using their defaults

statC = StatCollector.getInstance()

try:
    options = getopt.getopt(sys.argv[1:], "", ["numSimulations=", "numRepetitions=",
                            "randomSeed=", "dataRoot=", "explPolicy=", "lm=",
                            "explDecay=", "explMin=", "explStart=", "discountFactor=",
                            "learningRate=", "minReplay=", "minibatchSize=", "layers=",
                            "neuronsPerLayer=", "settingsFile="])

    options = options[0]
    numOptions = len(options)

    optionsProvided = [options[i][0] for i in range(numOptions)]
    currentOption = 0

    # This odd loop construct is done so that a settings file can be provided,
    # but options can be overwritten by specifying commands
    while (currentOption < numOptions):
        o = options[currentOption][0]
        a = options[currentOption][1]

        if o == "--settingsFile":
            fileSettings = readSettings(str(a))
            for newOption, newArgument in fileSettings:
                if newOption not in optionsProvided:
                    options.append([newOption, newArgument])

            numOptions = len(options)

        elif o == "--numSimulations":
            NUM_SIMULATIONS = int(a)
        elif o == "--numRepetitions":
            NUM_REPETITIONS = int(a)
        elif o == "--randomSeed":
            RANDOM_SEED = int(a)
        elif o == "--dataRoot":
            statC.setDataRoot(str(a))
        elif o == "--explPolicy":
            if (str(a).lower() == "boltzman"):
                EXPLORATION_POLICY = "boltzman"
            elif (str(a).lower() == "epsilon"):
                EXPLORATION_POLICY = "epsilon"
            else:
                print(f"Did not recognize exploration policy: {str(a)}.\nAborting...")
                exit(1)
        elif o == "--lm":
            if (str(a).lower() == "nn"):
                LOGIC_MODULE = "nn"
            elif (str(a).lower() == "tab"):
                LOGIC_MODULE = "tab"
            else:
                print(f"Did not recognize learning module: {str(a)}.\nAborting...")
                exit(1)
        elif o == "--explDecay":
            EPSILON_DECAY = float(a)
        elif o == "--explMin":
            MIN_EPSILON = float(a)
        elif o == "--explStart":
            START_EPSILON = float(a)
        elif o == "--discountFactor":
            DISCOUNT_FACTOR = float(a)
        elif o == "--learningRate":
            LEARNING_RATE = float(a)
        elif o == "--minReplay":
            MIN_REPLAY_MEMORY_SIZE = int(a)
        elif o == "--minibatchSize":
            MINIBATCH_SIZE = int(a)
        elif o == "--layers":
            LAYERS = int(a)
        elif o == "--neuronsPerLayer":
            NODES_IN_LAYER = int(a)

        currentOption += 1

except getopt.GetoptError as err:
    print(str(err))
    exit(1)

GlobalSettings.printMode = GlobalSettings.PRINT_NORMAL


#Setup code
np.random.seed(RANDOM_SEED)
env = Environment()

explPolicy = None
lm = None
agent = None

statC.startSession()
statC.addSessionData("random_seed", RANDOM_SEED)
statC.addSessionData("sims_per_run", NUM_SIMULATIONS)
statC.addSessionData("logic_module", LOGIC_MODULE)
statC.addSessionData("discount_factor", DISCOUNT_FACTOR)
statC.addSessionData("learning_rate", LEARNING_RATE)

if (LOGIC_MODULE == "nn"):
    statC.addSessionData("start_expl", START_EPSILON)
    statC.addSessionData("expl_decay", EPSILON_DECAY)
    statC.addSessionData("expl_policy", EXPLORATION_POLICY)
    statC.addSessionData("Model info", f"{LAYERS} x {NODES_IN_LAYER}, min replay:" +
                         f"{MIN_REPLAY_MEMORY_SIZE}, batch size: {MINIBATCH_SIZE}")


#The simulations themselves
for i in range(0, NUM_REPETITIONS):

    print(f"{cm.BACKED_C} {i} out of {NUM_REPETITIONS} simulations done.{cm.NORMAL}")


    statC.startRun()
    env.createRandomProblem()

    if (LOGIC_MODULE == "nn"):
        if (EXPLORATION_POLICY == "epsilon"):
            explPolicy = EpsilonGreedyPolicy(epsilon = START_EPSILON, decayRate = EPSILON_DECAY, minEpsilon = MIN_EPSILON)
        elif (EXPLORATION_POLICY == "boltzman"):
            explPolicy = BoltzmanExplorationPolicy(startingTemperature = START_EPSILON, temperatureDecay = EPSILON_DECAY, minTemperature = MIN_EPSILON)

        lm = QLearningNeuralModule( explorationPolicy = explPolicy, discountFactor = DISCOUNT_FACTOR, learningRate = LEARNING_RATE,
                                    minReplayMemorySize = MIN_REPLAY_MEMORY_SIZE, miniBatchSize = MINIBATCH_SIZE,
                                    layers = LAYERS, nodesInLayer = NODES_IN_LAYER)
    elif (LOGIC_MODULE == "tab"):
        lm = QLearningTabModule(explorationPolicy = GreedyPolicy(), discountFactor = 0, learningRate = 1)

    agent = Agent(env, lm)
    agent.train(NUM_SIMULATIONS)
