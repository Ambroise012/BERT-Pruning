import matplotlib.pyplot as plt
import numpy as np

sparity = (0, 0.6, 0.7, 0.8, 0.9)

model_size = {
    'CoLA':(85054464,33808185,25034823,17090692,8246372,4301466),
    'MNLI':(85054464,33684085,25285351,16678832,8213294,3918868),
    'MRPC':(85054464,33816042,25271161,16839280,8261770,4400991),
    'QNLI':(84722760,33697946,25024805,16370060,8504751,4340502),
    'QQP':(85054464,33673290,24974088,16608180,8593482,3917464),
    'RTE':(85054464,34047724,25300608,16887383,8283365,4100463),  
    'SST2':(85054464,33875521,24991393,16548011,8623494,4733899),
    'STSB':(85054464,33953119,25362184,16658322,8513841,4132657)
}
model_size_no_distill = {
    'SST2':(85054464,33790431,25658591,16107524,7522945),

    'MNLI':(85054464,33548405,25121821,16645710,8207443),
    'QQP':(85054464,33468285,24851476,16523702,7948811),

    # 'QQP with distill':(85054464,33673290,24974088,16608180,8593482,3917464),
    'QNLI':(85054464,33332450,23576822,15878056,7952449),

    # 'QNLI with distill':(85054464,33697946,25024805,16370060,8504751,4340502),
    # 'MRPC':(85054464,33816042,25271161,16839280,8261770,4400991),
    'MRPC':(85054464,31990631,23702592,14338013,1017160),
    # 'RTE with distill':(85054464,34047724,25300608,16887383,8283365,4100463),  
    'RTE':(85054464,30513776,16902375,10837228,233583),  
    # 'SST2 with distill':(85054464,33875521,24991393,16548011,8623494,4733899)
}

x = np.arange(len(sparity))  # the label locations
width = 0.1  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
for attribute, measurement in model_size_no_distill.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1
# ax.set_ylim(bottom=0, top=0.03)

ax.set_ylabel('Model size (bytes)')
ax.set_title('Pruning : Model size without distillation')
ax.set_xticks(x + width * len(model_size_no_distill) / 2, sparity)

# Placing the legend on the side
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('images/graph_model_size_no_distill.png', bbox_inches='tight')
plt.show()
