import matplotlib.pyplot as plt
import numpy as np

sparity = (0.1, 0.6, 0.8, 0.9, 0.95)

accuracy = {
    'QQP':(),
    'SST2':(46.7708402,26.28957,16.25215,11.24177,9.2),#ok
    'QNLI':(),
    # 'MNLI':(),
    'MRPC':(34.22431235531543,21.03471283150326,5.14063212116454,2.663924820912916,1.4348886455689411),# ok
    'RTE':(12.47463769,6.11704943,3.5869906129,1.674946,1.0852959),#ok
    'STSB':(80.93687,60.1658,45.5145054872,27.9758947,19.4759700), #ok
    'CoLA':(47.052743671,21.78709,13.3962164,9.5512071,6.8) #ok
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

plt.savefig('../../../../images/paper/no_prune/graph_accuracy.png', bbox_inches='tight')
plt.show()
