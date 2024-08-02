import matplotlib.pyplot as plt
import numpy as np

sparity = (0.1, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95)

accuracy = {
    'QQP':(),
    'SST2':(4.200452306401839,2.88570692779821,1.8378377917709701,1.513422424093299,1.1086279372556494,"""0.9, 0.95"""),
    'QNLI':(),
    # 'MNLI':(),
    'MRPC':(3.283173307176714,2.3257164332642767,1.6724291354633092,1.347162406141913,0.9174453031167895,0.4577380523347018,0.2825558493134899),
    'RTE':(3.2785632368867307,2.042709229541962,1.5954637898933112,1.2354784629744948,0.8230846281086064,0.5003550241547529,0.4),
    'STSB':(3.257635426103023,2.3686695182532596,1.645168195691025,1.3237539257919578,0.991834146934643,0.6277211507161459,0.27640377011215475),
    'CoLA':(3.311867886452945,2.2019239128156297,1.6392987401380652,1.3064581744995825,0.920224407519631,0.4919534278889264,0.2574166550419493)
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

ax.set_ylabel('Speedup (ms)')
ax.set_title('Accuracy - FT, A, DA')
ax.set_xticks(x + width * len(accuracy) / 2, sparity)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('../../../../images/paper/no_prune/graph_speedup.png', bbox_inches='tight')
plt.show()
