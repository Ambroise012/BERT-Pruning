import matplotlib.pyplot as plt
import numpy as np

sparity = ("FT", "A", "DA")

accuracy = {
    'QQP':(4.4,2.2189633138044737,2.215105367420394),
    'SST2':(4.7,2.1895513348623155,2.1910655095599114),
    'QNLI':(4.4,2.209695861079024,2.2072514378280402),
    # 'MNLI':(),
    'MRPC':(4.6,2.171568180385389,3.4082629424745),
    'RTE':(4.7,2.19267809456708,3.424580324155940),
    'STSB':(4.6,2.208854072972348,3.4119229316711425),
    'CoLA':(4.8,2.176251283668633,3.4102076279104625)
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

ax.set_ylabel('Time (ms)')
ax.set_title('Speedup - FT, A, DA')
ax.set_xticks(x + width * len(accuracy) / 2, sparity)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('../../../images/paper/no_prune/graph_speedup.png', bbox_inches='tight')
plt.show()
