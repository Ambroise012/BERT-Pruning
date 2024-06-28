import matplotlib.pyplot as plt
import numpy as np

sparity = (0.6, 0.7, 0.8, 0.9, 0.95)

# speed_up = {
#     'MNLI':(1.665,1.354,0.991,0.57082,0.385720),
#     'QQP':(1.815062,1.3926547,0.96457110,0.62336,0),
#     'QNLI':(2.0604,1.582409,1.07498,0.52307855,0.35623),
#     'SST2':(2.0450,1.51410,1.120,0.55990,0.30150),
#     # 'MRPC':(0.8675496,0.8188405,0.8190184,0.7893864,0.820031),
#     'RTE':(1.1380,0.82858,0.315549,0.1251207,0.0708645),  
# }
# training_time = {
#     'MNLI':(83157.24,82564.8,82955,81415.77,79538.19),
#     'QQP':(92976.07,92525.79,90517.8,90619.34,0),
#     'QNLI':(24167.53,23698.73,23256.47,22977.46,22945.54),
#     'SST2':(13528.98,13480.31,13296.8,13169.07,13052.21),
#     'MRPC':(3955.17,3920.92,3834.1,3952.25,4008.9),
#     'RTE':(2658.96,2692.68,2707.05,2778.94,2783.59),  
# }
training_time = {
    'MNLI':(83157.24,82564.8,82955,81415.77,79538.19),
    'QQP':(92976.07,92525.79,90517.8,90619.34,0),
    'QNLI':(24167.53,23698.73,23256.47,22977.46,22945.54),
    'SST2':(11705.5,13480.31,13296.8,13169.07,13052.21),
    'MRPC':(3955.17,3920.92,3834.1,3952.25,4008.9),
    'RTE':(2658.96,2692.68,2707.05,2778.94,2783.59),  
}


x = np.arange(len(sparity))  # the label locations
width = 0.1  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
for attribute, measurement in training_time.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1

ax.set_ylabel('Time (s)')
ax.set_title('Pruning - Adapter : Speedup')
ax.set_xlabel('Sparsity Ratio')
ax.set_xticks(x + width, sparity)
ax.legend()

plt.savefig('../../images/adapter/time_spar_train.png', bbox_inches='tight')
plt.show()
