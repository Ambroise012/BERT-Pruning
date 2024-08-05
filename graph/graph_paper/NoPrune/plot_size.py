import matplotlib.pyplot as plt
import numpy as np

sparity = ("FT", "A", "DA")

accuracy = {
    'QQP':(85054464,85054464,85054464),
    'SST2':(85054464,85054464,85054464),
    'QNLI':(85054464,85054464,85054464),
    # 'MNLI':(),
    'MRPC':(85054464,85054464,85054464),
    'RTE':(85054464,85054464,85054464),
    'STSB':(85054464,85054464,85054464),
    'CoLA':(85054464,85054464,85054464)
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

ax.set_ylabel('Size (params)')
ax.set_title('Model size - FT, A, DA')
ax.set_xticks(x + width * len(accuracy) / 2, sparity)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('../../../images/paper/no_prune/graph_size.png', bbox_inches='tight')
plt.show()
