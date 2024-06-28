import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("constantes.csv", header=None, skiprows=1)

plt.plot(df.index,df[0],label="lambda_1")
plt.plot(df.index,df[1],label="lambda_2")


# Add labels and legend
plt.xlabel('Iterations')
plt.ylabel('Lambda')
plt.title('Lambda 1, 2')
plt.legend()

plt.savefig('images/graph_lambda_all_last.png', bbox_inches='tight')

plt.show()
