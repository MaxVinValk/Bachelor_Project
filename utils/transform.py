def getAveragedAcc(results):
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
