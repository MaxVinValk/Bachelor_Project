'''

	This was an experiment that originally was inserted into the main of the project.
	As this was way too slow on my current device and I managed to find some parameters that worked
	reasonably well with manual selection, I decided to forego this approach.

	This code is really rough, as it was a proof of concept. I still elected to upload it to github,
	both as showing what I spent my time on, and perhaps using this later if I can get access to a faster GPU,
	refine the code, or both.

	Note that it is not-stand alone and does not function on its own, without the right imports.
'''

'''
#Genes: learningRate, minReplayMemorySize, miniBatchSize, layers, nodesInLayer
#Constraint: miniBatchSize <= minReplayMemorySize
geneRange = [[0.01, 1], [1, 200], [1, 200], [1, 16], [1, 32]]

GENE_POOL_SIZE = 64
NUM_REPEATED_RUNS = 1
NUM_RUNS = 64


ELITISM = 4
COPIED = 8
OFFSPRING = GENE_POOL_SIZE - COPIED
MUTATE_CHANCE = 0.05

def generateGene():
    gene = [0, []]

    gene[1] = [1, 32, 16, 2, 32]
    ' ''
    gene[1].append(np.random.uniform(geneRange[0][0], geneRange[0][1]))
    gene[1].append(np.random.randint(geneRange[1][0], geneRange[1][1]))

    miniBatch = np.random.randint(geneRange[2][0], geneRange[2][1])
    while miniBatch > gene[1][1]:
        miniBatch = np.random.randint(geneRange[2][0], geneRange[2][1])

    gene[1].append(miniBatch)
    gene[1].append(np.random.randint(geneRange[3][0], geneRange[3][1]))
    gene[1].append(np.random.randint(geneRange[4][0], geneRange[4][1]))
    '' '

    return gene


def mutateGene(gene):

    for i in range(0, len(gene[1])):
        if np.random.uniform(0, 1) <= MUTATE_CHANCE:

            if i == 0:
                change = np.random.uniform(geneRange[0][0], geneRange[0][1]) - geneRange[0][1]
                change /= 5
                gene[1][0] += change
                if (gene[1][0] > geneRange[0][1]):
                    gene[1][0] = geneRange[0][1]
                elif (gene[1][0] < geneRange[0][0]):
                    gene[1][0] = geneRange[0][0]
            elif i == 2:
                if (np.random.randint(0, 2) - 1):
                    gene[1][2] += 1
                else:
                    gene[1][2] -= 1

                if (gene[1][2] < geneRange[2][0]):
                    gene[1][2] = geneRange[2][0]
                elif (gene[1][2] > gene[1][1]):
                    gene[1][2] = gene[1][1]
            else:
                if (np.random.randint(0, 2) - 1):
                    gene[1][i] += np.random.randint(1, 4)
                else:
                    gene[1][i] -= np.random.randint(1, 4)

                    if (gene[1][i] < geneRange[i][0]):
                        gene[1][i] = geneRange[i][0]
                    elif (gene[1][i] >= geneRange[i][1]):
                        gene[1][i] = geneRange[i][1]

            ' ''
            if i == 0:
                gene[1][0] = np.random.uniform(geneRange[0][0], geneRange[0][1])
            elif i == 2:
                gene[1][2] = np.random.randint(geneRange[2][0], gene[1][1])
            else:
                gene[1][i] = np.random.randint(geneRange[i][0], geneRange[i][1])
            ' ''
    return gene





def createOffspring(firstParent, secondParent):
    firstChild = [0, []]
    secondChild = [0, []]

    geneLength = len(firstParent[1])

    splitPoint = np.random.randint(geneLength)

    firstChild[1] = firstParent[1][0:splitPoint] + secondParent[1][splitPoint:geneLength]
    secondChild[1] =secondParent[1][0:splitPoint] + firstParent[1][splitPoint:geneLength]

    firstChild = mutateGene(firstChild)
    secondChild = mutateGene(secondChild)

    return firstChild, secondChild

def selectGene(genePool, sum):

    total = np.random.uniform(0, sum)

    for i in range(0, len(genePool)):
        total -= genePool[i][0]

        if total <= 0:
            return i

    #should not get here


def getNextGeneration(genePool, run):
    genePool.sort(reverse = True)

    #calculate total score
    totalScore = 0
    for gene in genePool:
        totalScore += gene[0]

    print(f"{cm.BACKED_P}")
    print(f"{cm.BACKED_P}")
    print(f"{cm.NORMAL}")
    print(f"run {run}")
    print(f"average score: {totalScore / GENE_POOL_SIZE}")
    print(f"highest score: {genePool[0][0]}")
    print(f"{cm.BACKED_P}")
    print(f"{cm.BACKED_P}")
    print(f"{cm.NORMAL}")


    newPool = []

    for i in range(0, ELITISM):
        newPool.append(genePool[i])

    for i in range(0, COPIED):
        newPool.append(genePool[selectGene(genePool, totalScore)])

    for i in range(0, int(OFFSPRING/2)):
        firstParent = selectGene(genePool, totalScore)
        secondParent = selectGene(genePool, totalScore)

        while firstParent == secondParent:
            secondParent = selectGene(genePool, totalScore)

        firstChild, secondChild = createOffspring(genePool[firstParent], genePool[secondParent])
        newPool.append(firstChild)
        newPool.append(secondChild)

    return newPool


# To allow for easier reproducability
np.random.seed(1)


# Testing
env = Environment()
statC = StatCollector.getInstance()

#Issue: With dataOutput on true, then no data is returned by getStatisticLatest,
#as the data is saved at the end of the simulation...
#This is not an issue yet but should be noted for future development
statC.getSettings().disableDataOutput()
statC.startSession()


#generate initial gene pool
genePool = []
for i in range(0, GENE_POOL_SIZE):
    genePool.append(mutateGene(generateGene()))

runsPerformed = 0

while runsPerformed < NUM_RUNS:

    geneIdx = 0

    for gene in genePool:
        gene[0] = 0
        geneIdx += 1

        print(f"{cm.BACKED_C}")
        print(f"{cm.NORMAL}Gene {geneIdx}")

        for run in range(0, NUM_REPEATED_RUNS):

            statC.startRun()

            explPolicy = EpsilonGreedyPolicy(epsilon = 0.01, decayRate = 0, minEpsilon = 0.001)
            lm = QLearningNeuralModule(explorationPolicy = explPolicy, discountFactor = 0,
                    learningRate = gene[0], minReplayMemorySize = gene[1][1], miniBatchSize = gene[1][2],
                    layers = gene[1][3], nodesInLayer = gene[1][4])
            agent = Agent(env, lm)
            agent.train(NUM_SIMULATIONS)

            runAcc = statC.getStatisticLatest("Agent", "guessesAccuracyOverTime")
            learningOverall = sum(runAcc["data"]) / len(runAcc["data"])

            approxAcc = agent.evaluate(10000)

            gene[0] += approxAcc + learningOverall

        gene[0] /= NUM_REPEATED_RUNS

    #All genes evaluated



    genePool = getNextGeneration(genePool, runsPerformed)
    runsPerformed += 1



winnerIdx = np.argmax(geneScores)
print(f"\n\n{cm.INFO}Winner: {gene[winnerIdx]} {cm.NORMAL}")





'''

