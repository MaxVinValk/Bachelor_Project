import numpy as np

from ConsoleMessages import ConsoleMessages as cm
from Statistics import StatCollector, ClassCollector
from RunSettings import GlobalSettings
# Exploration modules

class ExplorationPolicy():

	cc = None

	def __init__(self):
		pass

	def getAction(self, qValues):
		pass


	def endSimulationUpdate(self):
		pass




class GreedyPolicy(ExplorationPolicy):

	def __init__(self):
		pass

	def getAction(self, qValues):
		return np.argmax(qValues)

	def endSimulationUpdate(self):
		pass


class EpsilonGreedyPolicy(ExplorationPolicy):

	#this function allows us to obtain an epsilon decay, which results in an epsilon of targetEpsilon at episode numEpisodes
	#under a multiplication epsilon decay
	@staticmethod
	def getDecay(targetEpsilon, numEpisodes, decayMode = "multiplication"):

		if (decayMode == "multiplication"):
			return np.power(np.e, (np.log(targetEpsilon) / numEpisodes))
		elif (decayMode == "linear"):
			return 1/numEpisodes

	#TODO turn into ENUM with string mapping
	VALID_DECAY_MODES = ["multiplication", "linear"]

	def __init__(self, epsilon, decayRate, minEpsilon, decayMode = "multiplication"):
		self.epsilon = epsilon
		self.decayRate = decayRate
		self.minEpsilon = minEpsilon


		if (decayMode not in self.VALID_DECAY_MODES):
			print(f"{cm.WARNING} Provided invalid decay mode {decayMode}. Defaulting to {self.VALID_DECAY_MODES[0]} instead.")
			decayMode = self.VALID_DECAY_MODES[0]
		self.decayMode = decayMode

		if GlobalSettings.printMode == GlobalSettings.PRINT_MODES[0]:
			print(f"{cm.NORMAL}Initialized eps-greedy exploration policy with start eps: {epsilon}, min eps: {minEpsilon}, decay mode: {decayMode}, and decay rate: {decayRate}")

		statC = StatCollector.getInstance()
		self.cc = statC.getClassCollector()
		self.cc.addStatistic("epsilonOverTime", "Epsilon value over time")

	def updateEpsilon(self):

		if (self.epsilon < self.minEpsilon):
			return

		if (self.decayMode is "multiplication"):
			self.epsilon = max(self.epsilon * self.decayRate, self.minEpsilon)

		elif (self.decayMode is "linear"):
			self.epsilon = max(self.epsilon - self.decayRate, self.minEpsilon)

	def endSimulationUpdate(self):
		statC = StatCollector.getInstance()
		self.cc.updateStatistic("epsilonOverTime", self.epsilon)

		self.updateEpsilon()

	def _takeRandom(self):

		if np.random.random() < self.epsilon:
			return True
		return False

	def getAction(self, qValues):
		if (self._takeRandom()):
			return np.random.randint(0, len(qValues))
		else:
			return np.argmax(qValues)

	def getEpsilon(self):
		return self.epsilon
