import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime


AVERAGE_OVER = 100_000

# TODO Change order of returns/train args to conform to the standard tuple description of a sample

class Agent():
	def __init__(self, environment, logicModule):
		self.environment = environment
		self.logicModule = logicModule

		stateDims, actionSize = environment.getEnvParams()

		logicModule.setupModule(stateDims, actionSize)

	def train(self, numSimulations):

		if (numSimulations % AVERAGE_OVER):
			print(f"\033[1;31mWARNING: number of simulations {numSimulations} " +
			f"is not divisible by average over {AVERAGE_OVER}." +
			f"Data may incomplete incorrect as a result.")

		rewardsOverTime = []
		epsilonOverTime = []
		guessesOverTime = []

		rewardsAvg = []
		epsilonAvg = []
		guessesAvg = []

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

			rewardsOverTime.append(collectiveReward)
			epsilonOverTime.append(self.logicModule.getExplorationPolicy().getEpsilon())
			guessesOverTime.append(guesses)


			if (episode % AVERAGE_OVER == 0):
				rewardsAvg.append(np.average(rewardsOverTime[-AVERAGE_OVER:]))
				epsilonAvg.append(np.average(epsilonOverTime[-AVERAGE_OVER:]))
				guessesAvg.append(np.average(guessesOverTime[-AVERAGE_OVER:]))


		plt.subplot(2, 2, 1)
		plt.plot(rewardsAvg)

		plt.title(f"Rewards received over time, averaged over {AVERAGE_OVER} runs")

		plt.subplot(2, 2, 2)
		plt.plot(epsilonAvg)

		plt.title(f"Epsilon over time, averaged over {AVERAGE_OVER} runs")

		plt.subplot(2, 2, 3)
		plt.plot(guessesAvg)

		plt.title(f"Guesses before success over time, averaged over {AVERAGE_OVER} runs")

		plt.subplot(2, 2, 4)
		plt.plot(1 / np.array(guessesAvg))

		plt.title(f"Classification precision, averaged over {AVERAGE_OVER} runs")

		plt.show()

		self.logicModule.saveTable(f"qTables/qTable_{datetime.now().strftime('%m-%d_%H:%M')}")
