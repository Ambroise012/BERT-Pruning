import matplotlib.pyplot as plt
import numpy as np

sparity = ("FT", "A", "DA")

accuracy = {
    'QQP':(2271.2093,2696.8929996543025,2212.591942127),
    'QNLI':(313.685,364.50146010188945,319.1705095277108),
    'SST2':(41.9448,58.343527282030664,46.78536163739),
    # 'MNLI':(),
    'MRPC':(20.0037,26.52198468061294,32.2266845423868),
    'RTE':(12.97,18.37970920268763,22.335429976975156),
    'STSB':(74.39,111.41054893145966,111.3582659371883),
    'CoLA':(64.84,67.57054476911937,85.38920276495992)
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

ax.set_ylabel('tCO2')
ax.set_title('Emissions - FT, A, DA')
ax.set_xticks(x + width * len(accuracy) / 2, sparity)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('../../../images/paper/no_prune/graph_emissions.png', bbox_inches='tight')
plt.show()
