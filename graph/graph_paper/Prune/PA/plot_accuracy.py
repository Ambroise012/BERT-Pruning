import matplotlib.pyplot as plt
import numpy as np

sparity = (0.1, 0.4, 0.6, 0.8, 0.9, 0.95)

accuracy = {
    'QQP':(91.00915162008409,90.52436309671036,89.85159534998763,70.8,66.465989,42.75162008409595,),
    'SST2':(90.59,89.44,88.53,87.84,86.5871559633027,84.40366972477065),#ok
    'QNLI':(90.46311550430166,89.09024345597657,87.58923668314114,86.16144975288303,85.321455,83.15943620721216),#ok
    # 'MNLI':(),
    'MRPC':(85.04901960784313,84.31372549019608,80.63725490196079,78.43137254901961,69.6078431372549,74.01960784313726),#accuracy
    'RTE':(67.14801444043321,62.09386281588448,62.09386281588448,62.8158844765343,60.64981949458483,55.96),#accuracy
    'STSB':(88.6360643157621,87.11357908681909,85.87818606380704,83.88758972529277,83.01862209010021,79.99258225885072),#comb score
    'CoLA':(55.95884617444483,54.97693861041112,51.00602849333835,45.648816752090837,33.961358161648586,20.843349206692793)#accuracy
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

ax.set_ylabel('Performance metrics')
ax.set_title('Efficiency - P + A')
ax.set_xticks(x + width * len(accuracy) / 2, sparity)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('../../../../images/paper/prune/PA/graph_accuracy.png', bbox_inches='tight')
plt.show()
