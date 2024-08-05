import matplotlib.pyplot as plt
import numpy as np

sparity = (0.1, 0.4, 0.6, 0.8, 0.9, 0.95)

accuracy = {
    'QQP':(3.99013219597963,2.6565,1.7975630749578143,1.1684984,0.65654,0.24984),
    'SST2':(4.200452306401839,2.88570692779821,1.8378377917709701,1.1086279372556494,0.27180647631303977,0.157519491440659),
    'QNLI':(3.947813912841529,2.7518857099319057,1.8457935643287754,0.9815237274672683,0.4254685,0.3351221782000711),
    # 'MNLI':(),
    'MRPC':(3.283173307176714,2.3257164332642767,1.6724291354633092,0.9174453031167895,0.4577380523347018,0.2825558493134899),
    'RTE':(3.2785632368867307,2.042709229541962,1.5954637898933112,0.8230846281086064,0.5003550241547529,0.4),
    'STSB':(3.257635426103023,2.3686695182532596,1.645168195691025,0.991834146934643,0.6277211507161459,0.27640377011215475),
    'CoLA':(3.311867886452945,2.2019239128156297,1.6392987401380652,0.920224407519631,0.4919534278889264,0.2574166550419493)
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
