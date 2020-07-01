#Processes a single line
def parseLine(line):

    name = "--"
    value = ""

    readName = True

    for char in line:
        if readName:
            if (char == " " or char == "\t"):
                readName = False
            else:
                name += char
        else:
            if (char not in [" ", "\t", "\n"]):
                value += char

    return [name, value]

#Reads in settings from a file and returns it in the correct format
def readSettings(file):
    settings = []

    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            if (len(line) > 0):
                settings.append(parseLine(line))
    return settings
