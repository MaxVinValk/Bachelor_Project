import numpy as np
import random
import os
import pickle


from LogicModules import QLearningNeuralModule
from ExplorationPolicies import EpsilonGreedyPolicy, BoltzmanExplorationPolicy
from operator import itemgetter
from datetime import datetime
from RunSettings import GlobalSettings
from ConsoleMessages import ConsoleMessages as cm

class GenePool():

    genePool = []
    currentGen = 0

    # To keep track of which run this is, for use in repeated runs in case of
    # an interruption during the simulation
    runCtr = 0

    GENE_POOL_SIZE = 64
    ELITISM = 8
    COPIED = 12
    OFFSPRING = GENE_POOL_SIZE - COPIED - ELITISM
    MUTATE_CHANCE = 0.05

    MUTATE_RANDOM = False

    MUTATE_CHANGE = 10

    #Sigma is used to scale mutation of continuous variables
    SIGMA = 0.1

    # Note: There is an additional constraint: miniBatchSize >= minReplayMemorySize

    #   For exploration policy selection, the following is relevant:
    #
    #       ExporationPolicy: Decides what policy to use
    #
    #       startingExplValue: (previously called startingEpsilon)
    #           In eps-greedy, determines starting epsilon
    #           In boltzman, starting temperature/1000 (so to get starting temp, this val * 1000)
    #
    #       decayRate:
    #           In eps-greedy + boltzman, determines the rate of decay
    #

    GENES = {   "learningRate" : {"type" : "float", "minValue" : 0.000001, "maxValue" : 1},
                "minReplayMemorySize" : {"type" : "int", "minValue" : 1, "maxValue" : 200},
                "miniBatchSize" : {"type" : "int", "minValue" : 1, "maxValue" : 200},
                "layers" : {"type" : "int", "minValue" : 1, "maxValue" : 16},
                "nodesInLayer" : {"type" : "int", "minValue" : 1, "maxValue" : 32},
                "startingExplValue" : {"type" : "float", "minValue" : 0.01, "maxValue" : 1},
                "decayRate" : {"type" : "float", "minValue" : 0.01, "maxValue" : 1},
                "explorationPolicy": {"type" : "select", "values" : ["epsilon", "boltzman"]}
            }

    # Used for output reasons
    lastWrittenTo = None

    # This method decodes a gene and turns it into a logic module instance

    @staticmethod
    def createLogicModule(gene):

        explPolicy = None

        if gene["explorationPolicy"] == "epsilon":
            explPolicy = EpsilonGreedyPolicy(epsilon = gene["startingExplValue"], decayRate = gene["decayRate"], minEpsilon = 0.01)
        elif gene["explorationPolicy"] == "boltzman":
            explPolicy = BoltzmanExplorationPolicy(startingTemperature = gene["startingExplValue"] * 1000,
                                                    temperatureDecay = gene["decayRate"], minTemperature = 10)

        return QLearningNeuralModule(explPolicy, 0, gene["learningRate"],
                                    gene["minReplayMemorySize"], gene["miniBatchSize"],
                                    gene["layers"], gene["nodesInLayer"])

    def __init__(self, genePoolSize = 64, elitism = 4, copied = 8, mutateChance = 0.05, limitExplTo = None, outputDir = "genepool"):

        if (limitExplTo == "epsilon"):
            self.GENES["explorationPolicy"]["values"] = ["epsilon"]
        elif (limitExplTo == "boltzman"):
            self.GENES["explorationPolicy"]["values"] = ["boltzman"]

        self.GENE_POOL_SIZE = genePoolSize
        self.ELITISM = elitism
        self.COPIED = copied
        self.OFFSPRING = self.GENE_POOL_SIZE - self.COPIED - self.ELITISM
        self.MUTATE_CHANCE = mutateChance
        self.outputDir = outputDir

        self.initRandom()

    # Fills the gene pool with random genes

    def initRandom(self):
        self.genePool = []
        for i in range(0, self.GENE_POOL_SIZE):
            self.genePool.append([0, self.randomGene()])


    # Takes the current genes with fitness and creates the next generation
    def evolve(self):
        self.genePool.sort(reverse = True, key = itemgetter(0))

        #Last stored was the intial generated genepool
        if self.lastWrittenTo is not None:
            with open(self.lastWrittenTo, "wb") as f:
                pickle.dump(self.genePool, f)

        newGenes = []

        #calculate total score:
        totalScore = 0
        for gene in self.genePool:
            totalScore += gene[0]

        if GlobalSettings.printMode == GlobalSettings.PRINT_PEREGRINE:
            print(f"[OWN_OUT] Generation info: best {self.genePool[0][0]}, sum: {totalScore}")


        for i in range(0, self.ELITISM):
            newGenes.append(self.genePool[i])

        for i in range(0, self.COPIED):
            selectedGene = self.genePool[self.selectGene(self.genePool, totalScore)]
            self.mutateGene(selectedGene[1])
            newGenes.append(selectedGene)

        for i in range(0, int(self.OFFSPRING/2)):
            firstParent = self.selectGene(self.genePool, totalScore)
            secondParent = self.selectGene(self.genePool, totalScore)

            while firstParent == secondParent:
                secondParent = self.selectGene(self.genePool, totalScore)

            firstChild, secondChild = self.createOffspring(self.genePool[firstParent], self.genePool[secondParent])
            newGenes.append(firstChild)
            newGenes.append(secondChild)

        #ensure all values are valid
        for gene in newGenes:
            self.enforceBounds(gene[1])

        self.genePool = newGenes
        self.currentGen += 1
        self.savePool()

    #Takes 2 parents, creates 2 children
    def createOffspring(self, firstParent, secondParent):
        firstChild = [0, {}]
        secondChild = [0, {}]

        geneLength = len(self.GENES)

        splitPoint = np.random.randint(geneLength)
        currentPoint = 0

        for key in sorted(self.GENES):

            if currentPoint < splitPoint:
                firstChild[1][key] = firstParent[1][key]
                secondChild[1][key] = secondParent[1][key]
            else:
                firstChild[1][key] = secondParent[1][key]
                secondChild[1][key] = firstParent[1][key]

            currentPoint += 1

        return firstChild, secondChild

    #Select gene from a genepool using roulette wheel selection
    def selectGene(self, genePool, sum):

        total = np.random.uniform(0, sum)

        for i in range(0, len(genePool)):
            total -= genePool[i][0]

            if total <= 0:
                return i

        #should not be reached but a failsafe
        return len(genePool) - 1

    #Applies mutation to the genes passed in, based on the chance to mutate each
    def mutateGene(self, gene):
        for key, value in gene.items():
            if np.random.uniform(0, 1) < self.MUTATE_CHANCE:
                geneInfo = self.GENES[key]

                if geneInfo["type"] == "float":
                    #gaussian
                    change = np.random.normal() * self.SIGMA
                    gene[key] = self.setInRange(gene[key] + change, geneInfo["minValue"], geneInfo["maxValue"])

                elif geneInfo["type"] == "int" or geneInfo["type"] == "select":
                    #select at random
                    self.setToRandomValue(gene, key, self.GENES[key])

    # Ensures that all values specified are valid w.r.t. additional constraints
    def enforceBounds(self, gene):
         #constraint: miniBatchSize <= minReplayMemorySize
         if (gene["miniBatchSize"] > gene["minReplayMemorySize"]):
             gene["miniBatchSize"] = gene["minReplayMemorySize"]

    # returns the value passed in bounded by the min and max
    def setInRange(self, value, min, max):
        if (value < min):
            return min
        elif (value > max):
            return max
        return value

    # Generates a random gene
    def randomGene(self):
        gene = {}

        for key in sorted(self.GENES):
            self.setToRandomValue(gene, key, self.GENES[key])

        return gene

    # Works only if values are being set front to back, if keys are sorted
    def setToRandomValue(self, gene, name, attributes):
        if attributes["type"] == "float":
            newValue = np.random.uniform(attributes["minValue"], attributes["maxValue"])
        elif attributes["type"] == "int":
            newValue =  np.random.randint(attributes["minValue"], attributes["maxValue"])
        elif attributes["type"] == "select":
            newValue = np.random.choice(attributes["values"])

        if (name == "miniBatchSize"):
            while (newValue > gene["minReplayMemorySize"]):
                newValue =  np.random.randint(attributes["minValue"], attributes["maxValue"])

        gene[name] = newValue

    #To call at the beginning of a run
    #assumes that the current runidx has not been started yet
    def startRun(self):
        os.mkdir(f"{self.outputDir}/run_{self.runCtr}")

    def endRun(self):
        self.runCtr += 1


    #deals with restarting the program if it has been interrupted
    def restart(self):

        if not os.path.exists(f"{self.outputDir}"):
            try:
                os.mkdir(f"{self.outputDir}")
            except OSError as e:
                print(f"{cm.WARNING} Failed to create directory for genepool data.{cm.NORMAL}")

            with open(f"{self.outputDir}/infoFile", "wb") as f:
                pickle.dump(self.currentGen, f)
                pickle.dump(self.GENE_POOL_SIZE, f)
                pickle.dump(self.ELITISM, f)
                pickle.dump(self.COPIED, f)
                pickle.dump(self.OFFSPRING, f)
                pickle.dump(self.MUTATE_CHANCE, f)

            return False
        else:
            self.loadPool()

            with open(f"{self.outputDir}/infoFile", "rb") as f:
                self.currentGen = pickle.load(f)
                self.GENE_POOL_SIZE = pickle.load(f)
                self.ELITISM = pickle.load(f)
                self.COPIED = pickle.load(f)
                self.OFFSPRING = pickle.load(f)
                self.MUTATE_CHANCE = pickle.load(f)

            if GlobalSettings.printMode == GlobalSettings.PRINT_PEREGRINE:
                print("[OWN_OUT] ####Restarting from file")

            return True

    def savePool(self):
        name = f"{self.outputDir}/run_{self.runCtr}/genes_gen_{self.currentGen}_{datetime.now().strftime('%m_%d - %H_%M_%S')}"
        self.lastWrittenTo = name

        with open(f"{self.outputDir}/infoFile", "wb") as f:
            pickle.dump(self.currentGen, f)
            pickle.dump(self.GENE_POOL_SIZE, f)
            pickle.dump(self.ELITISM, f)
            pickle.dump(self.COPIED, f)
            pickle.dump(self.OFFSPRING, f)
            pickle.dump(self.MUTATE_CHANCE, f)


        with open(name, "wb") as f:
            pickle.dump(self.genePool, f)

    def loadPool(self):

        #Get latest folder:
        latestNo = 0

        for file in os.listdir(f"{self.outputDir}"):
            if os.path.isfile(f"{self.outputDir}/{file}"):
                continue
            if "run_" in file:
                no = int(file[4:])
                if no > latestNo:
                    latestNo = no

        runFolder = f"{self.outputDir}/run_{latestNo}"
        self.runCtr = latestNo

        lastFile = None
        lastModification = 0

        for file in os.listdir(runFolder):

            if os.path.isdir(f"{runFolder}/{file}"):
                continue

            if "infoFile" == file:
                continue

            lmTemp = os.path.getmtime(f"{runFolder}/{file}")

            if (lmTemp > lastModification):
                lastModification = lmTemp
                lastFile = file
        print(f"lastFile: {lastFile}")
        if lastFile is not None:
            with open(f"{runFolder}/{lastFile}", "rb") as f:
                self.genePool = pickle.load(f)
            self.lastWrittenTo = f"{runFolder}/{lastFile}"
