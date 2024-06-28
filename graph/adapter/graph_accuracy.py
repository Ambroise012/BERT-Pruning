import matplotlib.pyplot as plt
import numpy as np

sparity = (0.6, 0.7, 0.8, 0.9, 0.95)

accuracy = {
    'QQP':(0.87467,0.8236,0.8525,0.8023,0),
    'SST2':(0.9025,0.8910,0.85779,0.83256,0.82110),
    'QNLI':(0.87607,0.86143,0.85227,0.82903,0.80724),
    'MNLI':(0.82312,0.81487,0.803769,0.778196,0.72654),
    'MRPC':(0.80392,0.75490,0.71078,0.7,0.6887254),
    'RTE':(0.60649,0.58483,0.57039,0.512635,0.5)
}

x = np.arange(len(sparity))  # the label locations
width = 0.1  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
for attribute, measurement in accuracy.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1
# ax.set_ylim(bottom=0, top=0.03)

ax.set_ylabel('Accuracy')
ax.set_title('Accuracy - Adapter - sparsity ratio of the pruning')
ax.set_xticks(x + width * len(accuracy) / 2, sparity)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('../../images/adapter/graph_accuracy.png', bbox_inches='tight')
plt.show()
