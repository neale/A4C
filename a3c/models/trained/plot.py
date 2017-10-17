import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

scores = []
x = []
with open('seaquest_a3c.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        print row
        scores.append(float(row[2]))
        x.append(float(row[1]))

plt.title("A3C Seaquest Mean Scores")
plt.xlabel("Training Steps")
plt.ylabel("Score")

plt.plot(x, scores)
plt.show()

