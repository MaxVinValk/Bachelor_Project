import pickle


def getResultsAyoobi(filePath):
	
	file_1 = open(filePath, 'r')
	file_1.contents = file_1.readlines()
	file_1.close()

	res = []


	for line in file_1.contents:
		if "iteration" in line:
			res.append([])
		else:
			curIt = len(res)-1
			line = line.replace('%', '')
			numbersFound = [int(s) for s in line.split() if s.isdigit()]
			if not len(numbersFound) == 1:
				print("Issue with parsing the following line:")
				print(line)
				print(f"{numbersFound} numbers found")
				exit(1)
			else:
				asFrac = numbersFound[0] / 100
				res[curIt].append(asFrac)
		
	return res


def getResultsSelf(filePath):
	res = []
	with open(filePath, "rb") as f:
		res = pickle.load(f)
	return res
	

		
		

