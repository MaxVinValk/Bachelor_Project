# Scriptie

This is meant to be a quick overview of the project in its current state, with
short descriptions of each file and usage information


## Useage

When running the program, the following arguments can be provided:
```
python3 Main.py [--numSimulations NUMBER] [--numRepetitions NUMBER] [--randomSeed SEED] [--dataRoot root]
```
Which allows one to change the number of simulations per run, the amount of runs, the
seed used for the RNG and the output folder for data respectively. 

For more in-depth changes, one can change the type of logic module and exploration policy within the code itself.
At this moment, the QLearningNeuralModule and QLearningTabularModule are functioning logic modules.
For exploration policies, Epsilon Greedy is supported as well as a greedy policy (used for evaluating the agent)

## Plotting the results

The StatCollector class can be used to load in and plot data. Usage is as follows,
from a python3 console:
```python
from Statistics.py import StatCollector
sc = StatCollector.getInstance()
sc.listFolders(dirName = None)
sc.load(FOLDER_NAME)
sc.loadLatest(dirName = None)

sc.plotRun(runIdx, averageOver = 1, shape = [2, 2], plotAll = True, *toPlot)
sc.plotAverage(averageOver = 1, shape = [2, 2], plotAll = True, *toPlot)
```

The listFolders function returns a list of the folders in the provided directory, or if
no argument is provided, the default output directory which can be viewed in the variable
dataRoot of the class StatCollector. loadLatest loads the latest run in the folder
provided, or if no folder is provided, the default output directory. The latest run is
determined by latest modification to any folder in the provided directorya.

With regards to the plotting functions, plotRun allows one to select 1 specific run to plot.
You can select it by providing as runIdx the run you want to plot.
The plotAverage function averages all statistics of all runs and outputs the plots of that.
The rest of the arguments that you can provide are the same:

averageOver is an argument that you can specify to average over an interval instead
of plotting each datapoint. This is especially useful when the amount of datapoints
collected gets really large. The shape provided is the shape of the resulting plots.
Data collected from separate classes will always be plotted on different graphs.
At most shape[0] * shape[1] subplots will be plotted on a single graph.
If one just wants to plot a (subset of) the data that has been collected, one can set plotAll
to false and provide as additional arguments the names of the specific statistics one wishes to plot.

## Getting the raw data
```python
r = sc.getRun(runIdx)
data = sc.getStatistic(run, className, statisticName)
runAvg = sc.averagedRun()
data2 = runAvg.getStatistic(className, statisticName)
```
getRun returns the run at the specified index.
If you want to have access to the stored data directly, one can get any statistic from any run (RunCollector instance)
with the getStatistic function. Provide the className that stores the statistic you want, and the name
of the desired statistic. If you want to get the averaged data, the runAvg function will get you
a RunCollector object, which also has a function allowing you to get any statistic in it.

## Information about loaded data

```python
sc.summarize(detailed = False)
sc.summarizeRun(runIdx)
sc.summarizeGivenRun(run)
```

The summarize function gives an overview of what data has been loaded in.
If detailed is set to true, it provides additional information on each run saved.
Using the summarizeRun function you can display information about an individual run by providing
its index.
Finally, using the summarizeGivenRun function, you can pass along a RunCollector object, and get
information on it. This is especially useful for when you want to get an overview of the averaged
run, as averagedRun() returns a RunCollector.

##Other
In case you want to merge multiple seperate sessions into 1, in the utils folder, there is a
script provided that can do that. It is advised to only do this if the runs are performed with different
seeds.

## Implementation details

### Files

#### Main
Entrypoint of the program

#### Environment
Holds all details with regards to the environment, such as states, actions, rewards
and which action is correct in each state. Can be given to an agent to interact with

#### Logic Module
A logic module is given to an agent. The logic module houses all decision making and
training functionality.
Currently the only implemented module is Q-learning.

#### ExplorationPolicy
Some logic modules use exploration policies to guide/dictate their decision making.
Currently the only exploration policy implemented is epsilon-greedy.

#### Agent
The agent can act within an environment and uses a logic module for choosing its actions.
It is the link between the environment and the logic module.

#### Statistics
Contains code for both collecting data from the program as well as plotting it.
More information in 'Collection of statistics via the StatCollector class' section.

#### Console Message
Contains some constants for console messages.


### Collection of statistics via the StatCollector class
The StatCollector class is responsible for collecting all data that we want to.
It manages all tracked data, is responsible for saving and loading the data, and
has plotting functionality. The StatCollector is a singleton class. For each class
that wishes to save data, it creates a ClassCollector object, responsible for storing
the information for that class. For now this suffices as we only have single instances
of each class. In the future, if we needed multiple instances to log information,
this maybe needs to be changed. After obtaining our class collector, we have to
register what data we want to record, by using the addStatistic method, providing a
name for the statistic as well as a title/description of the statistic. We then
add data to it with updateStatistic.

In short:
```python
statC = StatCollector.getInstance()
cc = statC.getClassCollector()

cc.addStatistic("statistic name", "statistic title")
cc.updateStatistic("statistic name", "value")
```

It is also possible to register data that is about the entire session in general, 
which can be done with addSessionData:

```python
statC = StatCollector.getInstance()
sc.addSessionData("name", value)
```

When using the StatCollector for logging data, the before saving any data at all, one
must call the startSession() function on the StatCollector instance. If you want to change
the output directory for the data, this has to be done prior to this call, because after
this call it will generate some additional folders if needed. Lastly, at the start of
each run, the function startRun() has to be called on the StatCollector.

###Utils/combiner.py
A utility file allowing the combination of multiple runs
