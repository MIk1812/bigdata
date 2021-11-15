import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def createBoxPlots():
    spread = np.random.rand(50) * 100
    center = np.ones(25) * 40
    flier_high = np.random.rand(10) * 100 + 100
    flier_low = np.random.rand(10) * -100
    d2 = np.concatenate((spread, center, flier_high, flier_low))
    d2 = np.concatenate((spread, center, flier_high, flier_low))
    print(d2)
    # Making a 2-D array only works if all the columns are the
    # same length.  If they are not, then use a list instead.
    # This is actually more efficient because boxplot converts
    # a 2-D array into a list of vectors internally anyway.
    fig, ax = plt.subplots()
    ax.boxplot(d2)
    plt.show()