import matplotlib.pyplot as plt
import numpy as np

sparity = (0, 0.6, 0.7, 0.8, 0.9, 0.95)

speed_up = {
    'MNLI':(3.904,0.956,0.724,0.471,0.248,0.185),
    'QQP':(3.94,0.956,0.728,0.456,0.332,0.26),
    'QNLI':(3.524,1.924,1.494,1.093,0.645,0.118),
    'SST2':(2.2,1.975,0.844, 0.611,0.391,0.232),
    'MRPC':(3.633,1.863,1.482,1.12,0.72,0.237),
    'RTE':(3.642,1.921,1.539,1.157,0.592,0.454),  
    'STSB':(3.525,1.933,1.5315,1.13,0.704,0.287)
}

model_size = {
    'CoLA':(84722760,33808185,25034823,17090692,8246372,4301466),
    'MNLI':(84722760,33684085,25285351,16678832,8213294,3918868),
    'MRPC':(84722760,33816042,25271161,16839280,8261770,4400991),
    'QNLI':(84722760,33697946,25024805,16370060,8504751,4340502),
    'QQP':(84722760,33673290,24974088,16608180,8593482,3917464),
    'RTE':(84722760,34047724,25300608,16887383,8283365,4100463),  
    'SST2':(84722760,33875521,24991393,16548011,8623494,4733899),
    'STSB':(84722760,33953119,25362184,16658322,8513841,4132657)
}

x = np.arange(len(sparity))  # the label locations
width = 0.1  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
for attribute, measurement in speed_up.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1

ax.set_ylabel('Time (ms)')
ax.set_title('Pruning : Speedup')
ax.set_xlabel('Sparsity Ratio')
ax.set_xticks(x + width, sparity)
ax.legend()

plt.savefig('time_spar.png', bbox_inches='tight')
plt.show()
