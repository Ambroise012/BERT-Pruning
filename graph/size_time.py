import matplotlib.pyplot as plt
import numpy as np
#ajouter la sparsité 0.95 et ajouter les remplir les bonnes valeurs dans les task
speed_up = {
    'MNLI':(3.904,0.956,0.724,0.471,0.248,0.185),
    'QQP':(3.94,0.956,0.728,0.456,0.332,0.26),
    'QNLI':(3.524,1.924,1.494,1.093,0.645,0.118),
    'SST2':(1.975,1.895,1.491,1.123,0.731,0.441),
    'MRPC':(3.633,1.863,1.482,1.12,0.72,0.237),
    'RTE':(3.642,1.921,1.539,1.157,0.592,0.454),  
    'STSB':(3.525,1.933,1.5315,1.13,0.704,0.287)
}

model_size = {
    'MNLI':(85054464,33684085,25285351,16678832,8213294,3918868),
    'QQP':(85054464,33673290,24974088,16608180,8593482,3917464),
    'QNLI':(85054464,33697946,25024805,16370060,8504751,4340502),
    'SST2':(85054464,33875521,24991393,16548011,8623494,4733899),
    'MRPC':(85054464,33816042,25271161,16839280,8261770,4400991),
    'RTE':(85054464,34047724,25300608,16887383,8283365,4100463),  
    'STSB':(85054464,33953119,25362184,16658322,8513841,4132657)
}

# Create a plot
fig, ax = plt.subplots(layout='constrained')

# Plot each task's model size as a function of speed up
for task in speed_up.keys():
    ax.plot(model_size[task], speed_up[task], label=task)

# Labeling the axes and title
ax.set_xlabel('Speedup (ms)')
ax.set_ylabel('Model Size (bytes)')
ax.set_title('Pruning: Model Size vs Speedup')

# Adding a legend
ax.legend()

# Save and show the plot
plt.savefig('images/time_size.png', bbox_inches='tight')
plt.show()
