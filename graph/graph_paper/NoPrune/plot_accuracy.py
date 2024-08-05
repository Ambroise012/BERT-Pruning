import matplotlib.pyplot as plt
import numpy as np

sparity = ("FT", "A", "DA")

accuracy = {
    'QQP':(87.68,90.902,56.381),
    'SST2':(92.32,91.39,51.26),
    'QNLI':(90.99,90.5,56.214),
    # 'MNLI':(),
    'MRPC':(86.03,81.8627,68.382),
    'RTE':(67.15,67.148,54.512),
    'STSB':(89.12,89.808,17.75),
    'CoLA':(59.57,60.583,2.1)
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
ax.set_title('Accuracy - FT, A, DA')
ax.set_xticks(x + width * len(accuracy) / 2, sparity)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('../../../images/paper/no_prune/graph_accuracy.png', bbox_inches='tight')
plt.show()
