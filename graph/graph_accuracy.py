import matplotlib.pyplot as plt
import numpy as np

sparity = (0.6, 0.7, 0.8, 0.9)

accuracy = {
    'QQP':( 0.907, 0.9064,0.9028,0.8907),
    'SST2':(0.9071,0.8933,0.8807,0.8739),
    'QNLI':(0.8783, 0.866,0.8468,0.8451),
    'MNLI':(0.8222,0.8229,0.8157,0.793),
    'MRPC':(0.8113,0.7647, 0.7353,0.6985),
    'RTE':(0.5523,0.556,0.5199,0.5271)
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
ax.set_title('Accuracy - sparsity ratio of the pruning')
ax.set_xticks(x + width * len(accuracy) / 2, sparity)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('images/graph_accuracy.png', bbox_inches='tight')
plt.show()
