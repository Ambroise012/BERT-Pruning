import matplotlib.pyplot as plt
import numpy as np

sparity = (0.6, 0.7, 0.8, 0.9, 0.95)

emission_inference = {
    'MNLI':(484.842,379.3614,299.646036,166.884,157.482),
    'QQP':(964.26131,753.9300,535.819,363.657,0),
    'QNLI':(132.2019685,107.9483,77.32281,43.6220,28.376),
    'SST2':(19.625,14.4312,10.6594,5.3279,2.4654),
    'MRPC':(8.02431,6.65194,4.27031297,1.52376,0.6974),
    'RTE':(6.27698,4.4679,3,1.19117,0.39047),  
}


x = np.arange(len(sparity))  # the label locations
width = 0.1  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
for attribute, measurement in emission_inference.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1
# ax.set_ylim(bottom=0, top=0.03)

ax.set_ylabel('Energy (mili J)')
ax.set_title('Pruning - Adapter : Energy consumption')
ax.set_xlabel('Sparsity Ratio')
ax.set_xticks(x + width, sparity)
ax.legend()
# plt.savefig('images/emissions_all.png', bbox_inches='tight')
plt.savefig('../../images/adapter/graph_energy.png', bbox_inches='tight')
plt.show()
