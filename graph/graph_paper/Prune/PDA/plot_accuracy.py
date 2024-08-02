import matplotlib.pyplot as plt
import numpy as np

sparsity = (0.1, 0.6, 0.8, 0.9, 0.95)

accuracy = {
    'QQP':(),
    'SST2':(50.11467889908257,50.34403669724771,49.08256880733945,49.08256880733945,47.256880733945), #ok
    'QNLI':(),
    # 'MNLI':(),
    'MRPC':(85.04901960784313,81.37254901960784,68.38235294117647,68.38235294117647,68.38235294117647),#ok
    'RTE':(46.20938628158845,46.93140794223827,47.29241877256318,47.29241877256318,47.29241877256318), # ok mais revoir les valeurs 
    'STSB':(50.07428267557028,46.33345259898447,33.07965448934107,29.8084794950969,24.655022790058848),
    'CoLA':(55.44150232282876,51.35883913087378,51.01325453983456,50.98463387782821,49.2) #ok
}

x = np.arange(len(sparsity))  # the label locations
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
ax.set_xticks(x + width * len(accuracy) / 2, sparsity)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('../../../../images/paper/no_prune/graph_accuracy.png', bbox_inches='tight')
plt.show()
