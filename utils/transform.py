def getAveragedAccOld(results):
	#To cumulative
	cumRes = []
	for i in range(0, len(results)):
		cumTemp = []
		counter = 0
		for j in range(0, len(results[i])):
			counter += results[i][j]
			cumTemp.append(counter)

		cumRes.append(cumTemp)

	#To cumulative averaged

	cumAvg = [0] * len(cumRes[i])
	for i in range(0, len(cumRes)):
		for j in range(0, len(cumRes[i])):
			cumAvg[j] += cumRes[i][j]

	for i in range(0, len(cumAvg)):
		cumAvg[i] /= len(cumRes)

	#To accuracy

	cumAcc = [0] * len(cumAvg)
	for i in range(0, len(cumAvg)):
		cumAcc[i] = (i+1) / cumAvg[i]

	return cumAcc


def convertGuessesToAccuracy(run):
	acc = []
	correct = 0

	for i in range(0, len(run)):
		if run[i] == 1:
			correct += 1
		acc.append(correct / (i+1))

	return acc

def convertSetOfRuns(runs):
	converted = []
	for run in runs:
		converted.append(convertGuessesToAccuracy(run))

	return converted

#Assumes all runs are of equal length
def averageRuns(runs):
	runLength = len(runs[0])
	avg = [0] * runLength

	for run in runs:
		if len(run) != runLength:
			print("Issue: Not all runs of the same length")
			return None

		for i in range(0, len(run)):
			avg[i] += run[i]

	for i in range(0, len(avg)):
		avg /= runLength

	return avg
