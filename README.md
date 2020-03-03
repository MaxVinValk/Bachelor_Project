# Scriptie

This is meant to be a quick overview of the project in its current state, with
short descriptions of each file and usage information


## Useage

In main, set the number of simulations you wish to run by setting NUM_SIMULATIONS.
Here you can also set what kind of logic module (a module housing all the decision making
logic) you want to pass to the agent, with what kind of exploration policy.
At the moment, only tabular q-learning is functional as logic module, and only
epsilon-greedy as exploration policy.

The output directory can be specified in Agent by setting SAVE_IN.

## Plotting the results

The StatCollector class can be used to load in and plot data. Usage is as follows,
from a python3 console:
´´´
from Statistics.py import StatCollector
sc = StatCollector.getInstance()
sc.load(FOLDER_NAME)
sc.plot(averageOver = 1, shape = [2, 2])
´´´

You can get information about the data held by using the summarize() function of
StatCollector. FOLDER_NAME is the folder holding all the data for a run.

averageOver is an argument that you can specify to average over an interval instead
of plotting each datapoint. This is especially useful when the amount of datapoints
collected gets really large. The shape provided is the shape of the resulting plots.
Data collected from separate classes will always be plotted on different graphs.
At most shape[0] * shape[1] subplots will be plotted on a single graph.

Further, there are some additional functions:

´´´
sc.summarize()
sc.plotClass(className, averageOver = 1, shape = [2, 2])
sc.plotStatistic(className, statName, averageOver = 1):
´´´

The summarize function gives an overview of what data has been loaded in.
The plotClass method allows one to plot all data from one class.
The plotStatistic method allows one to plot one statistic from a particular class

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
´´´
statC = StatCollector.getInstance()
cc = statC.getClassCollector()

cc.addStatistic("statistic name", "statistic title")
cc.updateStatistic("statistic name", "value")
´´´

After the program is done, use the save function of StatCollector and provide
a folder to hold the data.
