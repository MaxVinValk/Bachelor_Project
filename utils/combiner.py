import shutil, os


MAX_NUM_RUNS = 1000

PATH = os.getcwd()
OUTPUT = "fused"

currentFolder = 0

for folder in sorted(os.listdir(PATH)):
	for f2 in sorted(os.listdir(f"{PATH}/{folder}/rawData")):
		sourceFolder = f"{PATH}/{folder}/rawData/{f2}"
		
		if os.path.isfile(sourceFolder):
			continue		
		
		print(sourceFolder)
		if currentFolder >= MAX_NUM_RUNS:
			break
		else:
			newFolder = f"{PATH}/{OUTPUT}/rawData/run_{currentFolder}"
			
			shutil.copytree(sourceFolder, newFolder)
			currentFolder += 1
	
	if currentFolder >= MAX_NUM_RUNS:
		break

print("Fusion completed")
	
