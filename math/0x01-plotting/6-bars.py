#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

plt.title("Number of Fruit per Person")
plt.ylabel('Quantity of Fruit')
plt.ylim(0, 80, 10)
width = 0.5
labels = ['Farrah', 'Fred', 'Felicia']
plt.bar(labels, fruit[0], width, label='apples', color='r')
plt.bar(labels, fruit[1], width, label='bananas',
        bottom=fruit[0], color='yellow')
plt.bar(labels, fruit[2], width, label='oranges',
        bottom=fruit[0] + fruit[1], color='#ff8000')
plt.bar(labels, fruit[3], width, label='peaches',
        bottom=fruit[0] + fruit[1] + fruit[2], color='#ffe5b4')
legend = plt.legend(loc='upper right')
plt.show()
