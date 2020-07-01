import numpy as np

from ConsoleMessages import ConsoleMessages as cm
from Statistics import StatCollector, ClassCollector
from RunSettings import GlobalSettings

# Exploration modules
# Decides how the exploration/exploitation is addressed.

# Dummy class, meant to demonstrate which functions are necessary for a
# working exploration policy
class ExplorationPolicy():

	cc = None

	def __init__(self):
		pass

	#Should return which action to select from qValues
	def getAction(self, qValues):
		pass

	def endSimulationUpdate(self):
		pass



#Chooses the best possible action, always
class GreedyPolicy(ExplorationPolicy):

	def __init__(self):
		pass

	def getAction(self, qValues):
		return np.argmax(qValues)

	def endSimulationUpdate(self):
		pass

#Boltzman exploration
class BoltzmanExplorationPolicy(ExplorationPolicy):

	VALID_DECAY_MODES = ["multiplication", "linear"]

	def __init__(self, startingTemperature, temperatureDecay, minTemperature, decayMode = "multiplication"):

		self.temperature = startingTemperature
		self.temperatureDecay = temperatureDecay
		self.minTemperature = minTemperature

		if (decayMode not in self.VALID_DECAY_MODES):
			print(f"{cm.WARNING} Provided invalid decay mode {decayMode}. Defaulting to {self.VALID_DECAY_MODES[0]} instead.{cm.NORMAL}")
			decayMode = self.VALID_DECAY_MODES[0]
		self.decayMode = decayMode

	def getAction(self, qValues):
		proportion = [np.power(np.e, i/self.temperature) for i in qValues]
		sum = np.sum(proportion)

		proportion = [i/sum for i in proportion]

		for i in range(0, len(proportion)):
			if (np.isnan(proportion[i])):
				proportion[i] = 1

		selected = np.random.uniform(0, 1)

		for i in range(0, len(proportion)):
			selected -= proportion[i]

			if (selected <= 0):
				return i

	def endSimulationUpdate(self):
		if (self.decayMode is "multiplication"):
			self.temperature = max(self.temperature * self.temperatureDecay, self.minTemperature)

		elif (self.decayMode is "linear"):
			self.temperature = max(self.temperature - self.temperatureDecay, self.minTemperature)



class EpsilonGreedyPolicy(ExplorationPolicy):

	#this function allows us to obtain an epsilon decay, which results in an epsilon of targetEpsilon at episode numEpisodes
	#under a multiplication epsilon decay, assuming a starting epsilon of 1
	@staticmethod
	def getDecay(targetEpsilon, numEpisodes, decayMode = "multiplication"):

		if (decayMode == "multiplication"):
			return np.power(np.e, (np.log(targetEpsilon) / numEpisodes))
		elif (decayMode == "linear"):
			return 1/numEpisodes

	VALID_DECAY_MODES = ["multiplication", "linear"]


	def __init__(self, epsilon, decayRate, minEpsilon, decayMode = "multiplication"):
		self.epsilon = epsilon
		self.decayRate = decayRate
		self.minEpsilon = minEpsilon


		if (decayMode not in self.VALID_DECAY_MODES):
			print(f"{cm.WARNING} Provided invalid decay mode {decayMode}. Defaulting to {self.VALID_DECAY_MODES[0]} instead.")
			decayMode = self.VALID_DECAY_MODES[0]
		self.decayMode = decayMode

	def updateEpsilon(self):

		if (self.epsilon <= self.minEpsilon):
			return

		if (self.decayMode is "multiplication"):
			self.epsilon = max(self.epsilon * self.decayRate, self.minEpsilon)

		elif (self.decayMode is "linear"):
			self.epsilon = max(self.epsilon - self.decayRate, self.minEpsilon)

	def endSimulationUpdate(self):
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
