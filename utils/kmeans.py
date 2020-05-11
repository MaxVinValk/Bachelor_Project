import pickle
import os
import multiprocessing as mp

class Cluster():

    #Use a gene to init our intial location
    def __init__(self, gene):
        self.nearestGenes = []
        self.ownValue = gene

    def empty(self):
        self.nearestGenes = []

    def add(self, gene):
        self.nearestGenes.append(gene)

    def distance(self, gene):
        dist = 0
        for key, value in self.ownValue.items():
            dist += abs(value - gene[key])

        return dist

    def adjustCentre(self):
        #Calculate middle-point
        oldValue = self.ownValue

        if (len(self.nearestGenes) == 0):
            return 0

        newValue = self.nearestGenes[0]

        for i in range(1, len(self.nearestGenes)):
            for key, value in self.nearestGenes[i].items():
                newValue[key] += value

        change = 0

        for key, value in newValue.items():
            newValue[key] /= len(self.nearestGenes)
            change += abs(newValue[key] - oldValue[key])

        self.ownValue = newValue


        return change

    #Calculates the within-cluster sum of squares: Squared avg. distance to centre
    def WCSS(self):
        #Empty cluster guard, against a 0 division:
        if (len(self.nearestGenes) == 0):
            #print("Empty cluster found!")
            return 0

        wcss = 0

        #Calculate summed squared distance
        for gene in self.nearestGenes:
            for key, value in gene.items():
                dist = abs(self.ownValue[key] - gene[key])
                wcss += dist * dist


        #And average it
        return wcss / len(self.nearestGenes)


class MinHeap():

    def __init__(self, heapSize):
        self.heap = []
        self.heapSize = heapSize
        #0-Indexing
        self.heap.append(None)
        self.head = 1

        #init with low priority and no objects
        for i in range(0, heapSize):
            self.heap.append([0, None])


    def minHeapify(self, idx):
        #See if we are beyond the head
        if idx >= self.head:
            return
        leftIdx = idx * 2
        rightIdx = idx * 2 + 1
        lowest = idx

        if (leftIdx < self.head and self.heap[leftIdx][0] < self.heap[lowest][0]):
            lowest = leftIdx
        if (rightIdx < self.head and self.heap[rightIdx][0] < self.heap[lowest][0]):
            lowest = rightIdx

        if not lowest == idx:
            #swap idx with lowest
            tmp = self.heap[lowest]
            self.heap[lowest] = self.heap[idx]
            self.heap[idx] = tmp
            #call minheapify on the swapped index
            self.minHeapify(lowest)


    def process(self, priority, item):

        #If heap is not full: Just add it. We do the sorting after
        #The heap is entirely full to make this easier.
        if (self.head <= self.heapSize):
            self.heap[self.head][0] = priority
            self.heap[self.head][1] = item
            self.head += 1

            # If it is full with the last add, we do a heapsort:
            if (self.head > self.heapSize):
                for idx in range(int((self.head-1)/2), 0, -1):
                    self.minHeapify(idx)
        #If heap is full: See if it belongs
        elif (priority > self.heap[1][0]):
            self.heap[1][0] = priority
            self.heap[1][1] = item
            self.minHeapify(1)

    def getHeap(self):
        return self.heap



# IN: Root folder which contains the genepool folders of one or more runs
#       Each genepool folder contains 1 infoFile, and 1 or more run folders
#       Each run folder contains 1 or more genes files
#
#       numLoad represents the top number of results to load
#

def loadResults(rootFolder, numLoad):

    resultingHeap = MinHeap(numLoad)

    for folder in os.listdir(rootFolder):
        fpath = f"{rootFolder}/{folder}"

        if os.path.isfile(fpath):
            continue
        #Then go over run folders
        for run in os.listdir(fpath):

            rpath = f"{fpath}/{run}"

            #To deal with the infofile
            if os.path.isfile(rpath):
                continue

            for generation in os.listdir(rpath):

                genPath = f"{rpath}/{generation}"
                genes = []
                with open(genPath, "rb") as f:
                    genes = pickle.load(f)


                for gene in genes:
                    resultingHeap.process(gene[0], gene[1])
    return resultingHeap


#IN: a list of genes
#OUT: For each feature a histogram representing that feature

def getHists(genes, binSize = 0.05):
    numSteps = int(1 / binSize)
    hists = {'decayRate' : [0] * numSteps, 'layers' : [0] * 17, 'learningRate' : [0] * numSteps, 'minReplayMemorySize': [0] * 201,
    'miniBatchSize' : [0] * 201, 'nodesInLayer' : [0] * 33, 'startingEpsilon' : [0] * numSteps}

    for gene in genes:
        for key, value in gene[1].items():
            #Quick and dirty: floats from ints
            index = 0
            if key == 'startingEpsilon' or key == 'learningRate' or key == 'decayRate':
                index = int(value / binSize)
                #For the starting values exactly on the border
                if (index == numSteps):
                    index -= 1
            else:
                index = value
            hists[key][index] += 1
    return hists





def performClustering(genes, numClusters):
    if len(genes) < numClusters:
        print("Issue: Too many clusters specified for the amount of genes listed")
        exit(1)

    allClusters = []

    #To allow 1-indexing
    allClusters.append(None)

    print(f"Using {numClusters} clusters.")
    clusters = []
    for i in range(0, numClusters):
        clusters.append(Cluster(genes[i][1]))

    #do
    while (True):
        #Purge genes belonging to cluster
        for cluster in clusters:
            cluster.empty()

        #For each gene, find the nearest cluster and assign it to it
        for gene in genes:
            nearest = clusters[0].distance(gene[1])
            nearestCluster = clusters[0]

            for i in range(1, len(clusters)):
                newDist = clusters[i].distance(gene[1])
                if newDist < nearest:
                    nearest = newDist
                    nearestCluster = clusters[i]

            nearestCluster.add(gene[1])

        #Recalculate new centre
        change = 0
        for cluster in clusters:
            change += cluster.adjustCentre()

        #See if we reach stopping criterion
        if change < 0.0001:
            break

    #Calculate MSE of each cluster
    wcss = 0
    for cluster in clusters:
        wcss += cluster.WCSS()
        #We don't care about which members the cluster has
        cluster.empty()

    return wcss, clusters


def clusteringWorker(genes, inQueue, outArray):

    while (True):
        numClusters = None
        try:
            numClusters = inQueue.get(timeout = 1)
        except:
            break

        wcss, clusters = performClustering(genes, numClusters)
        outArray[numClusters] = wcss


def findK(genes, numWorkers):
    inQueue = mp.Queue()
    outArray = mp.Array('f', 4401, lock=False)

    wcssFile = "kmeans_wcss"

    #To make it feel like it goes faster I put it in reverse :)
    for i in range(4400, 0, -1):
        inQueue.put(i)

    processes = []

    for i in range(0, numWorkers):
        p = mp.Process(target = clusteringWorker, args=(genes, inQueue, outArray))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    with open(wcssFile, "wb") as f:
        pickle.dump(list(outArray), f)
