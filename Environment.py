import numpy as np

#TODO: Read these parameters from a configuration file

class Environment():
	
	# environment parameters
	COLORS = ["red", "blue", "yellow", "green"]
	TYPES = ["ball", "box", "person"]
	ACTIONS = [ ["continue", 2], ["push", 5], ["ask", 4], ["alt", 10] ]
	
	#Here we store for each combination of colors + types the action that resolves the situation
	BESTACTIONS = { "red ball"   : 1 , "red box"   : 3, "red person"   : 2, 
					"green ball" : 1 , "green box" : 3, "green person" : 2, 
					"blue ball" : 1 , "blue box" : 3, "blue person" : 3, 
					"yellow ball" : 1 , "yellow box" : 3, "yellow person" : 2,
					"none none" : 0
	}
	
	
	NO_FIELDS = 6
	
	#In the simulation only one field is relevant for choosing the correct behaviour
	RELEVANT_FIELD = 4
	
	SUCCESS_REWARD = 15
	FAILURE_PUNISHMENT = 5
	
	#The observed objects present
	fields = []
	
	def __init__(self):
		pass
	
	def createRandomState(self):
		self.fields.clear()
		
		for i in range(self.NO_FIELDS):
			
			#TODO: None, None
			
			obstacleCol = np.random.randint(0, len(self.COLORS))
			obstacleType = np.random.randint(0, len(self.TYPES))
			
			
			self.fields.append(obstacleCol)
			self.fields.append(obstacleType)
	
	def getState(self):
		return self.fields
	
	def getEnvParams(self):
		return [len(self.COLORS), len(self.TYPES)]*self.NO_FIELDS, len(self.ACTIONS)
	
	#TODO
	def performAction(self, action):
		
		currentState = self.fields
		
		#as we either do not move or end up in a final state, which makes the next state irrelevant
		nextState = self.fields
		
		#relevant field
		relevantObstacle = [ self.fields[self.RELEVANT_FIELD*2], self.fields[self.RELEVANT_FIELD*2 + 1] ]
		
		
		idx = self.COLORS[relevantObstacle[0]] + " " + self.TYPES[relevantObstacle[1]]
		
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
		self.createRandomState()
