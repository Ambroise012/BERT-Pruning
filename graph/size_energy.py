import matplotlib.pyplot as plt
import numpy as np
#ajouter la sparsité 0.95 et ajouter les remplir les bonnes valeurs dans les task
emission_inference = {
    'MNLI':(613.2763,247.1696,238.6993,142.6552,105.9484,86.3617),
    # 'QQP':(2513.1775,1294.3545,984.1864,672.0734,515.9815,489.6),
    'QNLI':(338.4177,173.7092,143.67,107.5581,75.0522,47.3255),
    'SST2':(54.1532,28.4318,22.9707,18.2162,12.8283,9.2956),
    'MRPC':(25.9012,13.6798,11.3028,8.6548,5.3752,4.0218),
    'RTE':(17.7073,9.7732,6.3144,5.21,4.0686,3.5331),  
    'STSB':(92.9982,50.4655,32.5371,32.1535,18.9664,12.7853)
}

model_size = {
    'MNLI':(84722760,33684085,25285351,16678832,8213294,3918868),
    # 'QQP':(84722760,33673290,24974088,16608180,8593482,3917464),
    'QNLI':(84722760,33697946,25024805,16370060,8504751,4340502),
    'SST2':(84722760,33875521,24991393,16548011,8623494,4733899),
    'MRPC':(84722760,33816042,25271161,16839280,8261770,4400991),
    'RTE':(84722760,34047724,25300608,16887383,8283365,4100463),  
    'STSB':(84722760,33953119,25362184,16658322,8513841,4132657)
}

# Create a plot
fig, ax = plt.subplots(layout='constrained')

# Plot each task's model size as a function of speed up
for task in emission_inference.keys():
    ax.plot(model_size[task], emission_inference[task], label=task)

# Labeling the axes and title
ax.set_ylabel('Emissions (mJ)')
ax.set_xlabel('Model Size (bytes)')
ax.set_title('Pruning: Model Size vs Energy Cost')

# Adding a legend
ax.legend()

# Save and show the plot
plt.savefig('images/size_emissions_rm_QQP.png', bbox_inches='tight')
plt.show()
