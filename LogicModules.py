import numpy as np

class LogicModule():

	def __init__(self, explorationPolicy):
		self.explorationPolicy = explorationPolicy

	def setupModule(self):
		pass

	def getAction(self, state):
		pass

	def train(self, origState, resState, action, reward, done):
		pass

	def endSimulationUpdate(self):
		pass

#TODO: Loading of a table from file

class QLearningTabModule(LogicModule):

	def __init__(self, explorationPolicy, discountFactor, learningRate, tableInFile = None, **kwargs):
		self.DISCOUNT_FACTOR = discountFactor
		self.LEARNING_RATE = learningRate
		self.tableInFile = tableInFile

		print(f"\033[0;0mInitialized Tabular Q-Learning with discount Factor: {discountFactor} and learning Rate: {learningRate}")


		super(QLearningTabModule, self).__init__(explorationPolicy)

	def setupModule(self, stateDims, actionSize):

		self.actionSize = actionSize;

		if (self.tableInFile == None):
			print(f"\033[0;0mSetting up Q-table of size:{stateDims + [actionSize]}")
			self.qTable = np.random.uniform(low=-2, high = 0, size = stateDims + [actionSize])
		else:
			print(f"Loading table: {self.tableInFile}")
			self.loadTable(self.tableInFile)

	def getAction(self, state):

		if (self.explorationPolicy.takeRandom()):
			return np.random.randint(0, self.actionSize)
		else:
			return np.argmax(self.qTable[tuple(state)])

	# TODO Perhaps add named tuples?
	def train(self, origState, resState, action, reward, done):

		# Get max future
		if done:
			maxFutureQValue = 0
		else:
			maxFutureQValue = np.max(tuple(resState))

		currentQValue = self.qTable[tuple(origState + [action])]

		self.qTable[tuple(origState + [action])] = currentQValue + self.LEARNING_RATE * (reward + self.DISCOUNT_FACTOR * maxFutureQValue)

	def endSimulationUpdate(self):
		self.explorationPolicy.endSimulationUpdate()

	def getExplorationPolicy(self):
		return self.explorationPolicy

	def loadTable(self, fileName):
		self.qTable = np.load(fileName)

	def saveTable(self, fileName):
		np.save(fileName, self.qTable)
