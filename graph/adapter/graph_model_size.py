import matplotlib.pyplot as plt
import numpy as np

sparity = (0, 0.6, 0.7, 0.8, 0.9, 0.95)

model_size = {
    'MNLI':(85054464,33480913,25170451,16599361,7860703,4331515),
    'MRPC':(85054464,32433476,25217137,14789410,4197715,1476688),
    'QNLI':(85054464,33467323,24739654,16328000,7632995,4102493),
    'QQP':(85054464,33103779,24639310,16179137,8398751,0),
    'RTE':(85054464,37018742,24559283,0,8516843,2587485),  
    'SST2':(85054464,33412423,23494337,16811687,7549785,3655706),
}


x = np.arange(len(sparity))  # the label locations
width = 0.1  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
for attribute, measurement in model_size.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1
# ax.set_ylim(bottom=0, top=0.03)

ax.set_ylabel('Model size (bytes)')
ax.set_title('Pruning - Adapter : Model size')
ax.set_xticks(x + width * len(model_size) / 2, sparity)

# Placing the legend on the side
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('../../images/adapter/graph_model_size_no_distill.png', bbox_inches='tight')
plt.show()
