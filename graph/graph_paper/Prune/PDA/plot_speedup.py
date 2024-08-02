import matplotlib.pyplot as plt
import numpy as np

sparity = (0.1, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95)

size = {
    'QQP':(),
    'SST2':(1.9689297320645885,1.0352106679470167,0.5721097145605524,0.37766302670907536,0.272302670907536),#ok
    'QNLI':(),
    # 'MNLI':(),
    'MRPC':(4.206484439326268,1.987176782944623,1.2113391983988853,0.7892912263348622,0.4393470976251813),#ok
    'RTE':(3.993594553587034,2.0576774587020568,1.2434098332321918,0.8254616110582622,0.48822109041933364),#ok
    'STSB':(1.9787427918952806,1.0526622387400846,0.8368908982527883,0.5977909774111029,0.36043778636999296),#ok
    'CoLA':()
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

ax.set_ylabel('Time')
ax.set_title('Speedup - P+D+A')
ax.set_xticks(x + width * len(size) / 2, sparity)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('../../../../images/paper/no_prune/graph_speedup.png', bbox_inches='tight')
plt.show()
