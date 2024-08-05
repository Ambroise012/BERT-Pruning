import matplotlib.pyplot as plt
import numpy as np

sparity = (0.1, 0.6, 0.8, 0.9)

size = {
    'QQP':(2.0437042459001526,0.982533986103172,0,0),
    'SST2':(1.9689297320645885,0.5721097145605524,0.37766302670907536,0.272302670907536),#ok
    'QNLI':(3.989732234922713,1.8229285003107676,1.0452005816492301,0.4941040771731565),
    # 'MNLI':(),
    'MRPC':(4.206484439326268,1.2113391983988853,0.7892912263348622,0.4393470976251813),#ok
    'RTE':(3.993594553587034,1.2434098332321918,0.8254616110582622,0.48822109041933364),#ok
    'STSB':(1.9787427918952806,0.8368908982527883,0.5977909774111029,0.36043778636999296),#ok
    'CoLA':(1.9824828606847256,1.0158996405897223,0.5815955051026154,0.37790934084378164)#ok
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

ax.set_ylabel('Time (ms)')
ax.set_title('Speedup - P+D+A')
ax.set_xticks(x + width * len(size) / 2, sparity)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('../../../../images/paper/prune/PDA/graph_speedup.png', bbox_inches='tight')
plt.show()
