import matplotlib.pyplot as plt

def plotResults(avgEps, avgBoltz, avgEps, avgABL):

    plt.plot(avgEps, "-.")
    plt.plot(avgBoltz, "--") ; plt.plot(avgTab, "-") ; plt.plot(avgABL, ":")
    plt.grid(True)
    plt.ylim(0, 1)
    plt.legend(["Approx. based Q-learning (epsilon-greedy)", "Approx. based Q-learning (Boltzman)", "Tabular Q-learning (greedy)", "ABL"])
    plt.xlabel("Number of Recovery Attempts")
    plt.ylabel("Classification Precision Percentage")
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
    plt.show()
