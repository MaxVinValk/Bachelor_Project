import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from Statistics import StatCollector, ClassCollector
from ConsoleMessages import ConsoleMessages as cm

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

		statC = StatCollector.getInstance()

		cc = statC.getClassCollector()

		cc.addStatistic("rewardsOverTime", "Rewards received over time")
		cc.addStatistic("guessesOverTime", "Number of guesses over time")
		cc.addStatistic("guessesAccuracyOverTime", "Accuracy of guesses over time")

		for episode in tqdm(range(1, numSimulations+1), ascii=True, unit="simulation"):
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

			cc.updateStatistic("rewardsOverTime", collectiveReward)
			cc.updateStatistic("guessesOverTime", guesses)
			cc.updateStatistic("guessesAccuracyOverTime", 1 / guesses)

		#date = datetime.now().strftime('%m-%d_%H-%M')

		'''
		if (not os.path.exists(MODEL_LOCATION)):
			try:
				os.mkdir(MODEL_LOCATION)
			except OSError:
				print(f"{cm.WARNING} Failed to create directory for data. Data may be lost...")

		if (not os.path.exists(DATA_LOCATION)):
			try:
				os.mkdir(DATA_LOCATION)
			except OSError:
				print(f"{cm.WARNING} Failed to create directory for model. Model may be lost...")
		'''

		statC.save()
		#self.logicModule.save(MODEL_LOCATION + f"/qTable_{date}")

	def getAction(self, state):
		return self.logicModule.getAction(state)
