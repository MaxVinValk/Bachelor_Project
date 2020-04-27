import os
import getopt
import sys
import pickle
from operator import itemgetter

rootFolder = None
top = 10

try:
	options = getopt.getopt(sys.argv[1:], "", ["root=", "top="])
	
	for o, a in options[0]:
		if o == "--root":
			rootFolder = str(a)
		elif o == "--top":
			top = int(a)

except getopt.GetoptError as err:
	print(str(err))
	exit(1)

if rootFolder == None:
	print("No folder specified. Aborting")
	exit(1)
elif not os.path.isdir(rootFolder):
	print(f"location named is not a directory: {rootFolder}. Aborting")
	exit(1)

res = []

for i in range(0, top):
	res.append([-1, {}, ""])



for folder in os.listdir(rootFolder):
	folderPath = f"{rootFolder}/{folder}"
	if os.path.isdir(folderPath):
		for run in os.listdir(folderPath):
			filePath = f"{folderPath}/{run}"
			with open(filePath, "rb") as f:
				data = pickle.load(f)

				for gene in data:
					if gene[0] > res[0][0]:
						res[0] = gene + [filePath]
						res.sort(key = itemgetter(0))

for i in range(top -1, -1, -1):
	print(res[i])
			
