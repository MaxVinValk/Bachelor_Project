import numpy as np

from tqdm import tqdm
from datetime import datetime
from Statistics import StatCollector
from ConsoleMessages import ConsoleMessages as cm

# TODO Change order of returns/train args to conform to the standard tuple description of a sample

class Agent():
	def __init__(self, environment, logicModule):
		self.environment = environment
		self.logicModule = logicModule

		stateDims, actionSize = environment.getEnvParams()

		logicModule.setupModule(stateDims, actionSize)

	def train(self, numSimulations):
		statC = StatCollector.getInstance();

		statC.addStatistic("rewardsOverTime", "Rewards received over time")
		statC.addStatistic("guessesOverTime", "Number of guesses over time")
		statC.addStatistic("guessesAccuracyOverTime", "Accuracy of guesses over time")

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

			statC.updateStatistic("rewardsOverTime", collectiveReward)
			statC.updateStatistic("guessesOverTime", guesses)
			statC.updateStatistic("guessesAccuracyOverTime", 1 / guesses)

		date = datetime.now().strftime('%m-%d_%H:%M')

		statC.saveData(f"data/results_{date}")
		self.logicModule.saveTable(f"qTables/qTable_{date}")
