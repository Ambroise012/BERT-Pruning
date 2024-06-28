import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("loss.csv", header=None, skiprows=1)

# Transpose the DataFrame to have the correct orientation for plotting
df_transposed = df.T

# Plot each column
# for i in range(df_transposed.shape[1]):
#     plt.plot(df_transposed.index, df_transposed[i], label=f'Column {i+1}')

plt.plot(df.index,df[0],label="loss")
plt.plot(df.index,df[1],label="lagrangian loss")
plt.plot(df.index,df[2],label="distill loss")
plt.plot(df.index,df[3],label="ce distill")


# Add labels and legend
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Titre')
plt.legend()

plt.savefig('images/graph_all_loss_last.png', bbox_inches='tight')
# plt.savefig('graph_lagrangian_loss.png', bbox_inches='tight')
# plt.savefig('graph_distill_loss.png', bbox_inches='tight')



plt.show()
