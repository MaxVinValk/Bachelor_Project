import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from Statistics import StatCollector, ClassCollector
from ConsoleMessages import ConsoleMessages as cm
from ExplorationPolicies import GreedyPolicy
from RunSettings import GlobalSettings

class Agent():

	# To initialize an agent, it needs to be given an environment in which
	# it operates and a logic module, which decides how it takes actions in the environment
	def __init__(self, environment, logicModule):
		self.environment = environment
		self.logicModule = logicModule

		stateDims, actionSize = environment.getEnvParams()

		logicModule.setupModule(stateDims, actionSize)

	# Performs a training for a set number of simulations
	def train(self, numSimulations):

		statC = StatCollector.getInstance()
		cc = statC.getClassCollector()

		cc.addStatistic("guessesOverTime", "Number of guesses over time");

		# Toggles between the pre-generated training set of states, and the evaluation
		# set of states
		self.environment.setUseTrain(True)
		totalGuesses = 0
		firstCorrect = 0

		#Ensuring a progress bar is only shown when running normally
		disableProgress = GlobalSettings.printMode != GlobalSettings.PRINT_NORMAL

		for episode in tqdm(range(1, numSimulations+1), ascii=True, unit="simulation", disable = disableProgress):
			self.environment.reset()
			done = False

			guesses = 0

			while not done:
				state = self.environment.getState()
				action = self.logicModule.getAction(state)
				origState, resState, reward, done =  self.environment.performAction(action)
				guesses += 1
				self.logicModule.train(origState, resState, action, reward, done)

			self.logicModule.endSimulationUpdate()

			cc.updateStatistic("guessesOverTime", guesses)

			totalGuesses += guesses

			if guesses == 1:
				firstCorrect += 1

		statC.save()
		return firstCorrect / numSimulations

	def getAction(self, state):
		return self.logicModule.getAction(state)

	def evaluate(self, numberOfRuns):
		# Swap to the evaluation state set.
		self.environment.setUseTrain(False)
		# first we swap out our exploration policy in the logic module to rate current performance
		# of the learned behaviour, without the exploration policy interfering
		originalExplorationPolicy = self.logicModule.getExplorationPolicy()
		self.logicModule.setExplorationPolicy(GreedyPolicy())

		score = 0

		#The reset env will move to the next untried evaluation/validation state
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
