import csv
import numpy as np


timeStep = []
series = []

with open("./Sunspots.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    next(reader)
    for row in reader:
        timeStep.append(int(row[0]))
        series.append(float(row[2]))

timeStep = np.array(timeStep)
series = np.array(series)
trainTime, valTime = timeStep[:int(len(timeStep) * 0.8)], timeStep[int(len(timeStep) * 0.8):]
trainX, valX = series[:int(len(series) * 0.8)], series[int(len(series) * 0.8):]