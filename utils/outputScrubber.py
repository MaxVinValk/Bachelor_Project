def filterOutput(filename):

    resOutput = []
    identifyingPhrase = "[OWN_OUT]"
    phraseLength = len(identifyingPhrase)

    with open(filename, "r") as f:
        t = f.readline()

        while (bool(t)):
            if (identifyingPhrase in t):
                resOutput.append(t[phraseLength:].strip())

            t = f.readline()

    return resOutput

def getProgressOverTime(rawOutput):

    runs = []
    for sentence in rawOutput:

        if "Restarting from file" in sentence or "Performing simulation 0" in sentence:
            runs.append([[], []])

        if "Generation info" in sentence:
            bestStart = sentence.find("best ") + len("best ")
            bestEnd = sentence.find(",")
            sumStart = sentence.find("sum: ") + len("sum: ")

            runs[-1][0].append(float(sentence[bestStart:bestEnd]))
            runs[-1][1].append(float(sentence[sumStart:]))

    return runs
