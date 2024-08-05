import matplotlib.pyplot as plt
import numpy as np

sparity = (0.1, 0.6, 0.8, 0.9)

size = {
    'QQP':(76426035,33511554,0,0),
    'SST2':(76049304,34097540,16384751,8215074), #ok
    'QNLI':(76562344,33648114,16662863,8424932),
    # 'MNLI':(),
    'MRPC':(76540882,32932376,16539959,8416474), #ok
    'RTE':(76354113,32805867,16636990,8320087),#ok
    'STSB':(76063863,33768329,16552541,8592128), #ok
    'CoLA':(76198069,33519092,16572287,8778579) #ok
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

ax.set_ylabel('Size (params)')
ax.set_title('Model size - P+D+A')
ax.set_xticks(x + width * len(size) / 2, sparity)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('../../../../images/paper/prune/PDA/graph_size.png', bbox_inches='tight')
plt.show()
