import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from Statistics import StatCollector, ClassCollector
from ConsoleMessages import ConsoleMessages as cm
from ExplorationPolicies import GreedyPolicy
from RunSettings import GlobalSettings

SAVE_IN = "/media/max/88B5-59E2/data"
MODEL_LOCATION = SAVE_IN + "/" + "qTable"
DATA_LOCATION = SAVE_IN + "/" + "rawData"

class Agent():
	def __init__(self, environment, logicModule):
		self.environment = environment
		self.logicModule = logicModule

		stateDims, actionSize = environment.getEnvParams()

		logicModule.setupModule(stateDims, actionSize)

	def train(self, numSimulations):

		#statC = StatCollector.getInstance()
		#cc = statC.getClassCollector()

		#cc.addStatistic("rewardsOverTime", "Rewards received over time")
		#cc.addStatistic("guessesOverTime", "Number of guesses over time")
		#cc.addStatistic("guessesAccuracyOverTime", "Accuracy of guesses over time")

		self.environment.setUseTrain(True)
		runningAccuracy = 0
		disableProgress = GlobalSettings.printMode == GlobalSettings.PRINT_MODES[1]

		for episode in tqdm(range(1, numSimulations+1), ascii=True, unit="simulation", disable = disableProgress):
			self.environment.reset()
			done = False

			collectiveReward = 0
			guesses = 0

			while not done:
				state = self.environment.getState()
				action = self.logicModule.getAction(state)
				origState, resState, reward, done =  self.environment.performAction(action)

				self.logicModule.train(origState, resState, action, reward, done)
				collectiveReward += reward
				guesses += 1

			self.logicModule.endSimulationUpdate()

			#cc.updateStatistic("rewardsOverTime", collectiveReward)
			#cc.updateStatistic("guessesOverTime", guesses)

			acc = 1 / guesses
			#cc.updateStatistic("guessesAccuracyOverTime", 1 / guesses)

			runningAccuracy += acc / numSimulations

		#statC.save()
		return runningAccuracy

	def getAction(self, state):
		return self.logicModule.getAction(state)

	def evaluate(self, numberOfRuns):
		self.environment.setUseTrain(False)
		#first we swap out our exploration policy in the logic module to rate current performance
		#of the learned behaviour
		originalExplorationPolicy = self.logicModule.getExplorationPolicy()
		self.logicModule.setExplorationPolicy(GreedyPolicy())

		score = 0

		#Generate numberOfRuns random simulations
		for i in range(0, numberOfRuns):
			self.environment.reset()
			state = self.environment.getState()
			action = self.logicModule.getAction(state)
			origState, resState, reward, done = self.environment.performAction(action)

			if done:
				score += 1

		#Give back the old exploration policy
		self.logicModule.setExplorationPolicy(originalExplorationPolicy)

		return score / numberOfRuns
