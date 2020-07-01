import shutil, os


def combineRuns(rootFolder, maxNumRuns = 1000):
	output = "fused"

	currentFolder = 0

	for folder in sorted(os.listdir(rootFolder)):
		for f2 in sorted(os.listdir(f"{rootFolder}/{folder}/rawData")):
			sourceFolder = f"{rootFolder}/{folder}/rawData/{f2}"

			if os.path.isfile(sourceFolder):
				continue

			print(sourceFolder)
			if currentFolder >= maxNumRuns:
				break
			else:
				newFolder = f"{rootFolder}/{output}/rawData/run_{currentFolder}"

				shutil.copytree(sourceFolder, newFolder)
				currentFolder += 1

		if currentFolder >= maxNumRuns:
			break

	print("Fusion completed")
