import numpy as np
from RunSettings import GlobalSettings
from ConsoleMessages import ConsoleMessages as cm


class Environment():

	# environment parameters
	COLORS = ["red", "blue", "yellow", "green", "none"]
	TYPES = ["ball", "box", "person", "none"]
	ACTIONS = [ ["continue", 2], ["push", 5], ["ask", 4], ["alt", 10] ]

	#Here we store for each combination of colors + types the action that resolves the situation
	BESTACTIONS = {}

	NO_FIELDS = 6

	#In the simulation only one field is relevant for choosing the correct behaviour
	RELEVANT_FIELD = 4

	SUCCESS_REWARD = 30
	FAILURE_PUNISHMENT = 10

	#A flag which, when disabled, uses a memory of generated cases.
	#This is relevant for when we are comparing multiple agents on performance
	USE_SETS = False

	#When use sets is set, use train is used to determine if the validation or training
	#set should be used.
	USE_TRAIN = False
	trainSet = []
	trainIndex = 0
	validationSet = []
	validationIndex = 0

	#The observed objects present
	fields = []

	def __init__(self):
		#self._createRandomProblem()
		pass

	#Used to generate a training and validation set which is used for genetic algorithms
	def initSets(self, lengthTrain, lengthValidation):

		self.USE_SETS = True

		self.trainIndex = 0
		self.validationIndex = 0
		self.trainSet = []
		self.validationSet = []

		for i in range(0, lengthTrain):
			self.trainSet.append(self._createRandomState())

		for i in range(0, lengthValidation):
			self.validationSet.append(self._createRandomState())

	def setUseTrain(self, value):
		self.USE_TRAIN = value

	def createRandomProblem(self):

		self.BESTACTIONS = {}

		for i in range(0, len(self.COLORS) - 1):
			for j in range(0, len(self.TYPES) - 1):
				self.BESTACTIONS[f"{self.COLORS[i]} {self.TYPES[j]}"] = np.random.randint(len(self.ACTIONS))

		self.BESTACTIONS["none none"] = np.random.randint(len(self.ACTIONS))


	#Allows an agent to be tested against all possible scenarios
	def testAgainstAll(self, agent):

		currentState = [0, 0] * self.NO_FIELDS
		noColors = len(self.COLORS)
		noTypes = len(self.TYPES)

		#TODO: Give agent temporarily a greedy only module
		score = 0
		guesses = 0
		while True:

			#do the evaluation of current environment
			action = agent.getAction(currentState)
			relevantIdx = self.COLORS[currentState[2*self.RELEVANT_FIELD]] + " " + self.TYPES[currentState[2*self.RELEVANT_FIELD + 1]]


			if self.BESTACTIONS[relevantIdx] == action:
				score += 1
			guesses += 1

			#update environment

			currentField = 0

			while currentField < self.NO_FIELDS:
				if (currentState[currentField*2] == noColors - 1 and currentState[currentField*2 + 1] == noTypes - 1):
					#cannot update, need to increase currentField
					currentState[currentField*2] = 0
					currentState[currentField*2 + 1] = 0
					currentField += 1
				else:

					if (currentState[currentField * 2] == noColors - 2 and currentState[currentField * 2  + 1] == noTypes - 2):
							currentState[currentField*2] = noColors - 1
							currentState[currentField*2 + 1] = noTypes - 1
					elif (currentState[currentField * 2 + 1] < noTypes - 2):
						currentState[currentField * 2  + 1] += 1
					else:
						currentState[currentField * 2] += 1
						currentState[currentField * 2 + 1] = 0

					break

			if currentField >= self.NO_FIELDS:
				break

		if (GlobalSettings.printMode == GlobalSettings.PRINT_NORMAL):
			print(f"accuracy: {score/guesses}")




	def _createRandomState(self):
		#self.fields.clear()
		newProblem = []

		for i in range(self.NO_FIELDS):

			# Accounting for the none-none type. Should be generated in the right proportion
			if (np.random.randint(0, (len(self.COLORS) -1) * (len(self.TYPES) -1) + 1) == 0):
				obstacleCol = len(self.COLORS) - 1
				obstacleType = len(self.TYPES) - 1
			else:
				obstacleCol = np.random.randint(0, len(self.COLORS) - 1)
				obstacleType = np.random.randint(0, len(self.TYPES) - 1)

			newProblem.append(obstacleCol)
			newProblem.append(obstacleType)

		return newProblem

	def getState(self):
		return self.fields

	# +1 for the NONE NONE state
	def getEnvParams(self):
		return [len(self.COLORS), len(self.TYPES)]*self.NO_FIELDS, len(self.ACTIONS)

	def performAction(self, action):

		currentState = self.fields

		#as we either do not move or end up in a final state, which makes the next state irrelevant
		nextState = self.fields

		#relevant field
		relevantObstacle = [ self.fields[self.RELEVANT_FIELD*2], self.fields[self.RELEVANT_FIELD*2 + 1] ]


		idx = self.COLORS[relevantObstacle[0]] + " " + self.TYPES[relevantObstacle[1]]

		#print(f"Action to perform: {self.BESTACTIONS[idx]}, received: {action}")

		if self.BESTACTIONS[idx] == action:
			reward = self.SUCCESS_REWARD
			done = True
		else:
			reward = -self.FAILURE_PUNISHMENT
			done = False

		# include action cost
		reward = reward - self.ACTIONS[action][1]

		return currentState, nextState, reward, done

	def reset(self):
		if (not self.USE_SETS):
			self.fields = self._createRandomState()
		else:
			if self.USE_TRAIN:
				self.fields = self.trainSet[self.trainIndex]
				self.trainIndex += 1
			else:
				self.fields = self.validationSet[self.validationIndex]
				self.validationIndex += 1

	def resetSetIndexes(self):
		self.trainIndex = 0
		self.validationIndex = 0
