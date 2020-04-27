import numpy as np
import random
import os
import pickle


from LogicModules import QLearningNeuralModule
from ExplorationPolicies import EpsilonGreedyPolicy
from operator import itemgetter
from datetime import datetime
from RunSettings import GlobalSettings
from ConsoleMessages import ConsoleMessages as cm

'''

    USAGE in a main:
        create GenePool

    for however many simulation loops:
        get the genePool

        for each gene in the genePool:
            perform 200 simulations
            score performance by running a number of simulations
            store performance with the genepool

            evolve the genepool

    output the results to a file






    TODO: Range is inclusive in SetToRange but exclusive in the random
    functions.

    Use np.nextafter(x, y) to fix this, where x is the value you want the next
    after one of and y indicates the direction

'''



class GenePool():

    genePool = []
    currentGen = 0

    #To keep track of which run this is, for use in repeated runs
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

    #Note: There is an additional constraint: miniBatchSize >= minReplayMemorySize
    #TODO: Can't we remove the value type and use instanceof(numbers.Real) ?
    GENES = {   "learningRate" : {"type" : "float", "minValue" : 0.0001, "maxValue" : 1},
                "minReplayMemorySize" : {"type" : "int", "minValue" : 1, "maxValue" : 200},
                "miniBatchSize" : {"type" : "int", "minValue" : 1, "maxValue" : 200},
                "layers" : {"type" : "int", "minValue" : 1, "maxValue" : 16},
                "nodesInLayer" : {"type" : "int", "minValue" : 1, "maxValue" : 32}
            }

    def createLogicModule(gene):
        #TODO: Encode expl. policy params ?
        decay = EpsilonGreedyPolicy.getDecay(targetEpsilon = 0.01, numEpisodes = 100)

        explPolicy = EpsilonGreedyPolicy(epsilon = 1, decayRate = decay, minEpsilon = 0.01)
        return QLearningNeuralModule(explPolicy, 0, gene["learningRate"],
                                    gene["minReplayMemorySize"], gene["miniBatchSize"],
                                    gene["layers"], gene["nodesInLayer"])

    def __init__(self, genePoolSize = 64, elitism = 4, copied = 8, mutateChance = 0.05):

        self.GENE_POOL_SIZE = genePoolSize
        self.ELITISM = elitism
        self.COPIED = copied
        self.OFFSPRING = self.GENE_POOL_SIZE - self.COPIED - self.ELITISM
        self.MUTATE_CHANCE = mutateChance

        self.initRandom()

    def initRandom(self):
        self.genePool = []
        for i in range(0, self.GENE_POOL_SIZE):
            self.genePool.append([0, self.randomGene()])


    #employing changes:
    # gaussian noise in mutation
    # only mutation for the copied parents
    # reproduce using arithmic crossover
    def evolveNew(self):
        self.genePool.sort(reverse = True, key = itemgetter(0))

        newGenes = []

        #calculate total score:
        totalScore = 0
        for gene in self.genePool:
            totalScore += gene[0]

        if GlobalSettings.printMode == GlobalSettings.PRINT_MODES[1]:
            print(f"[OWN_OUT] Generation info: best {self.genePool[0][0]}, sum: {totalScore}")

        for i in range(0, self.ELITISM):
            newGenes.append(self.genePool[i])

        for i in range(0, self.COPIED):
            selectedGene = self.genePool[self.selectGene(self.genePool, totalScore)]
            self.mutateGeneNew(selectedGene[1])
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

    def evolve(self):
        self.genePool.sort(reverse = True, key = itemgetter(0))

        newGenes = []

        #calculate total score:
        totalScore = 0
        for gene in self.genePool:
            totalScore += gene[0]

        if GlobalSettings.printMode == GlobalSettings.PRINT_MODES[1]:
            print(f"[OWN_OUT] Generation info: best {self.genePool[0][0]}, sum: {totalScore}")

        for i in range(0, self.ELITISM):
            newGenes.append(self.genePool[i])

        for i in range(0, self.COPIED):
            newGenes.append(self.genePool[self.selectGene(self.genePool, totalScore)])

        for i in range(0, int(self.OFFSPRING/2)):
            firstParent = self.selectGene(self.genePool, totalScore)
            secondParent = self.selectGene(self.genePool, totalScore)

            while firstParent == secondParent:
                secondParent = self.selectGene(self.genePool, totalScore)

            firstChild, secondChild = self.createOffspring(self.genePool[firstParent], self.genePool[secondParent])
            newGenes.append(firstChild)
            newGenes.append(secondChild)

        for gene in newGenes:
            self.mutateGene(gene[1])
            self.enforceBounds(gene[1])

        self.genePool = newGenes
        self.currentGen += 1
        self.savePool()

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


    def selectGene(self, genePool, sum):

        total = np.random.uniform(0, sum)

        for i in range(0, len(genePool)):
            total -= genePool[i][0]

            if total <= 0:
                return i

        #should not be reached but a failsafe
        return len(genePool) - 1

    def mutateGeneNew(self, gene):
        for key, value in gene.items():
            if np.random.uniform(0, 1) < self.MUTATE_CHANCE:
                geneInfo = self.GENES[key]

                if geneInfo["type"] == "float":
                    #gaussian
                    change = np.random.normal() * self.SIGMA
                    gene[key] = self.setInRange(gene[key] + change, geneInfo["minValue"], geneInfo["maxValue"])

                elif geneInfo["type"] == "int":
                    #select at random
                    self.setToRandomValue(gene, key, self.GENES[key])


    def mutateGene(self, gene):

        #key is the property name, value the numerical value
        for key, value in gene.items():
            if np.random.uniform(0, 1) < self.MUTATE_CHANCE:
                if self.MUTATE_RANDOM:
                    self.setToRandomValue(gene, key, value)
                else:
                    #obtain the information about the gene such as its datatype and
                    geneInfo = self.GENES[key]

                    if geneInfo["type"] == "float":
                        mutateBy = np.random.uniform(0, (geneInfo["maxValue"] - geneInfo["minValue"]) / self.MUTATE_CHANGE)

                    elif geneInfo["type"] == "int":
                        mutateBy = np.random.randint(0, int((geneInfo["maxValue"] - geneInfo["minValue"]) / self.MUTATE_CHANGE))

                    #To center the change so it can be negative
                    mutateBy -= ((geneInfo["maxValue"] - geneInfo["minValue"]) / self.MUTATE_CHANGE) / 2

                    if (geneInfo["type"] == "int"):
                        mutateBy = int(mutateBy)

                    #We update the new value and make certain it is in the correct range
                    gene[key] = self.setInRange(gene[key] + mutateBy, geneInfo["minValue"], geneInfo["maxValue"])

    def enforceBounds(self, gene):
         #There is an additional constraint: miniBatchSize <= minReplayMemorySize
         for key, value in gene.items():
             if key == "miniBatchSize":
                 if (value > gene["minReplayMemorySize"]):
                     gene[key] = gene["minReplayMemorySize"]


    def setInRange(self, value, min, max):
        if (value < min):
            return min
        elif (value > max):
            return max
        return value

    def randomGene(self):
        gene = {}

        for key in sorted(self.GENES):
            self.setToRandomValue(gene, key, self.GENES[key])

        return gene

    #Works only if values are being set front to back, if keys are sorted
    def setToRandomValue(self, gene, name, attributes):
        if attributes["type"] == "float":
            newValue = np.random.uniform(attributes["minValue"], attributes["maxValue"])
        elif attributes["type"] == "int":
            newValue =  np.random.randint(attributes["minValue"], attributes["maxValue"])

        if (name == "miniBatchSize"):
            while (newValue > gene["minReplayMemorySize"]):
                newValue =  np.random.randint(attributes["minValue"], attributes["maxValue"])

        gene[name] = newValue

    #To call at the beginning of a run
    #assumes that the current runidx has not been started yet
    def startRun(self):
        os.mkdir(f"genepool/run_{self.runCtr}")

    def endRun(self):
        self.runCtr += 1


    #deals with restarting the program if it has been interrupted
    def restart(self):

        if not os.path.exists("genepool"):
            try:
                os.mkdir("genepool")
            except OSError as e:
                print(f"{cm.WARNING} Failed to create directory for genepool data.{cm.NORMAL}")

            with open("genepool/infoFile", "wb") as f:
                pickle.dump(self.currentGen, f)
                pickle.dump(self.GENE_POOL_SIZE, f)
                pickle.dump(self.ELITISM, f)
                pickle.dump(self.COPIED, f)
                pickle.dump(self.OFFSPRING, f)
                pickle.dump(self.MUTATE_CHANCE, f)

            #self.savePool()
            return False
        else:
            self.loadPool()

            with open("genepool/infoFile", "rb") as f:
                self.currentGen = pickle.load(f)
                self.GENE_POOL_SIZE = pickle.load(f)
                self.ELITISM = pickle.load(f)
                self.COPIED = pickle.load(f)
                self.OFFSPRING = pickle.load(f)
                self.MUTATE_CHANCE = pickle.load(f)

            if GlobalSettings.printMode == GlobalSettings.PRINT_MODES[1]:
                print("[OWN_OUT] ####Restarting from file")

            return True

    def savePool(self):
        name = f"genepool/run_{self.runCtr}/genes_gen_{self.currentGen}_{datetime.now().strftime('%m_%d - %H_%M_%S')}"

        with open("genepool/infoFile", "wb") as f:
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

        for file in os.listdir("genepool"):
            if os.path.isfile(f"genepool/{file}"):
                continue
            if "run_" in file:
                no = int(file[4:])
                if no > latestNo:
                    latestNo = no

        runFolder = f"genepool/run_{latestNo}"
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
