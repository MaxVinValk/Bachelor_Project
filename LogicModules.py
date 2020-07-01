# The implementation of the logic modules was based on code provided by 
# Harrison Kinsley of http://www.pythonprogramming.net

import numpy as np
import random
import time

from ConsoleMessages import ConsoleMessages as cm
from Statistics import StatCollector
from RunSettings import GlobalSettings

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

from ModTensorBoard import ModifiedTensorBoard

from collections import deque

# template for what is needed in a logic module

class LogicModule():

	def __init__(self, explorationPolicy):
		self.explorationPolicy = explorationPolicy

	def setupModule(self, stateDims, actionSize):
		pass

	def getAction(self, state):
		pass

	def train(self, origState, resState, action, reward, done):
		pass

	def endSimulationUpdate(self):
		pass

	def getExplorationPolicy(self):
		return self.explorationPolicy

	def save(self, fileName):
		pass

	def load(self, fileName):
		pass

	def getExplorationPolicy(self):
		return self.explorationPolicy

	def setExplorationPolicy(self, explorationPolicy):
		self.explorationPolicy = explorationPolicy


class QLearningTabModule(LogicModule):

	def __init__(self, explorationPolicy, discountFactor, learningRate, tableInFile = None, **kwargs):
		self.DISCOUNT_FACTOR = discountFactor
		self.LEARNING_RATE = learningRate
		self.tableInFile = tableInFile

		super(QLearningTabModule, self).__init__(explorationPolicy)

	def setupModule(self, stateDims, actionSize):

		self.actionSize = actionSize;

		if (self.tableInFile == None):
			self.qTable = np.random.uniform(low=-2, high = 0, size = stateDims + [actionSize])
		else:
			print(f"Loading table: {self.tableInFile}")
			self.loadTable(self.tableInFile)

	def getAction(self, state):
		return self.explorationPolicy.getAction(self.qTable[tuple(state)])


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

	def load(self, fileName):
		self.qTable = np.load(fileName)

	def save(self, fileName):
		np.save(fileName, self.qTable)



class QLearningNeuralModule(LogicModule):

	REPLAY_MEMORY_SIZE = 50_000
	MIN_REPLAY_MEMORY_SIZE = 32
	MINIBATCH_SIZE = 16

	ONE_HOT_ENCODING = True

	LAYERS = 2
	NODES_IN_LAYER = 32


	def __init__(self, explorationPolicy, discountFactor, learningRate, minReplayMemorySize = 32, miniBatchSize = 16, layers = 2, nodesInLayer = 32):
		self.explorationPolicy = explorationPolicy
		self.DISCOUNT_FACTOR = discountFactor
		self.LEARNING_RATE = learningRate

		self.targetUpdateCounter = 0

		self.MIN_REPLAY_MEMORY_SIZE = minReplayMemorySize
		self.MINIBATCH_SIZE = miniBatchSize
		self.LAYERS = layers
		self.NODES_IN_LAYER = nodesInLayer

		random.seed(1)


	def setupModule(self, stateDims, actionSize):

		self.LEARNING_RATE = 0.02798532247789199
		self.MIN_REPLAY_MEMORY_SIZE = 123
		self.MINIBATCH_SIZE = 88
		self.NODES_IN_LAYER = 26

		self.stateDims = np.array(stateDims)

		self.model = self._createModel(stateDims, actionSize)

		self.replayMemory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
		self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{int(time.time())}")

		self.targetUpdateCounter = 0


	def _createModel(self, stateDims, actionSize):

		if self.ONE_HOT_ENCODING:
			inSize = sum(stateDims)
		else:
			inSize = len(stateDims)

		model = Sequential()

		model.add(Dense(self.NODES_IN_LAYER, input_dim = inSize))
		model.add(Activation("elu"))

		for i in range(0, self.LAYERS - 1):
			model.add(Dense(self.NODES_IN_LAYER))
			model.add(Activation("elu"))


		model.add(Dense(actionSize))
		model.add(Activation("linear"))

		model.compile(loss="mse", optimizer=Adam(lr=self.LEARNING_RATE), metrics=['accuracy'])

		return model


	def _normalizeState(self, state):

		if self.ONE_HOT_ENCODING:
			newState = [0] * int(sum(self.stateDims))

			currentIdx = 0

			for idx in range(0, len(state)):

				checkIdx = currentIdx + state[idx]
				newState[checkIdx] = 1
				currentIdx += self.stateDims[idx]

			return newState

		else:
			return state / self.stateDims

	def getAction(self, state):
		normalizedState = np.array([self._normalizeState(state)])

		values = self.model.predict(normalizedState)[0]
		return self.explorationPolicy.getAction(values)

	def train(self, origState, resState, action, reward, done):

		currentExperience = (self._normalizeState(origState), self._normalizeState(resState), action, reward, done)

		self.replayMemory.append(currentExperience)

		#Check if we have enough memories
		if len(self.replayMemory) < self.MIN_REPLAY_MEMORY_SIZE:
			return

		#This is bugtesting:
		if (self.MINIBATCH_SIZE > self.MIN_REPLAY_MEMORY_SIZE):
			print(f"{cm.WARNING}Minibatch size:\t\t{self.MINIBATCH_SIZE}")
			print(f"{cm.WARNING}replaymemory size:\t\t{self.MIN_REPLAY_MEMORY_SIZE}")
			print(f"{cm.WARNING}in memory current::\t\t {len(self.replayMemory)}{cm.NORMAL}")

		#this may cause the current batch to contain the current experience twice.
		miniBatch = random.sample(self.replayMemory, self.MINIBATCH_SIZE-1)
		miniBatch.append(currentExperience)

		currentStates = np.array([transition[0] for transition in miniBatch])
		currentQScores = self.model.predict(currentStates)

		#NO NEED FOR THIS WITH DISCOUNT_FACTOR OF 0
		if (self.DISCOUNT_FACTOR != 0):
			resStates = np.array([transition[1] for transition in miniBatch])
			futureQScores = self.targetModel.predict(resStates)

		X = []
		y = []

		for index, (batchOrigState, batchResState, batchAction, batchReward, batchDone) in enumerate(miniBatch):

			#NO NEED FOR THIS WITH DISCOUNT_FACTOR OF 0

			if (self.DISCOUNT_FACTOR != 0):
				if not batchDone:
					maxFutureQ = np.max(futureQScores[index])
					newQ = batchReward + self.DISCOUNT_FACTOR * maxFutureQ
				else:
					newQ = batchReward
			else:
				newQ = batchReward

			currentQ = currentQScores[index]
			currentQ[batchAction] = newQ

			X.append(currentStates[index])
			y.append(currentQ)

		# Fit on all samples as one batch, log only on terminal state
		self.model.fit(np.array(X), np.array(y), batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=None)


	def endSimulationUpdate(self):
		self.explorationPolicy.endSimulationUpdate()


	def save(self, fileName):
		pass

	def load(self, fileName):
		pass
