import matplotlib.pyplot as plt
import numpy as np

sparity = (0.1, 0.4, 0.6, 0.8, 0.9, 0.95)

accuracy = {
    'QQP':(2078.9687937029407,1542.5656,963.828857519133,589.54456,452.1687,402.656),
    'SST2':(39.17721438,26.9496512,16.8624486,10.502339531,9.770133224,6.9510065634),
    'QNLI':(283.56186,199.088059,135.111217,73.39452,42.875694,28.838254),
    # 'MNLI':(),
    'MRPC':(31.3110027,22.50445,20.7822,9.56997,6.7217714,5.897119),
    'RTE':(24.93704,17.2751,13.39366,10.86743,6.40852,4.525180),
    'STSB':(119.2969,83.788756,74.8788,49.558,34.70340,17.1950),
    'CoLA':(98.04916,66.62620,50.89722,25.28965,15.80160,14.160490)
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
ax.set_title('Emissions - PA')
ax.set_xticks(x + width * len(accuracy) / 2, sparity)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('../../../../images/paper/no_prune/graph_emissions.png', bbox_inches='tight')
plt.show()
