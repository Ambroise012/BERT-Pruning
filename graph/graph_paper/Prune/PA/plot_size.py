import matplotlib.pyplot as plt
import numpy as np

sparity = (0.1, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95)

size = {
    'QQP':(),
    'SST2':(77069251,52424818,32304631,25334745,17336240,7736329,3706722),
    'QNLI':(),
    # 'MNLI':(),
    'MRPC':(76137194,49633995,32627579,25615665,15797688,6586497,3564516),
    'RTE':(75855888,43576024,32952590,25016131,14225660,7697194,6277504),
    'STSB':(76448137,51940896,32608487,24223997,15988398,8092964,3230479),
    'CoLA':(78171071,49746788,34299499,25549896,16538757,7764607,3743225)
}

x = np.arange(len(sparity))  # the label locations
width = 0.1  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
for attribute, measurement in size.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1
# ax.set_ylim(bottom=0, top=0.03)

ax.set_ylabel('size (Mb)')
ax.set_title('Model size - PA')
ax.set_xticks(x + width * len(size) / 2, sparity)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('../../../../images/paper/no_prune/PA/graph_size.png', bbox_inches='tight')
plt.show()
