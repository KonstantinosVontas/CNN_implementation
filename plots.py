from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import csv
import unidecode as unidecode
from torch import flatten




df_50k_cnn = pd.read_csv('./plots_for_thesis/50k_LayerNorm_Celux2_Linear_end_3000_epochs_All_Losses_Errors_batch_128.csv')
df_100k_cnn = pd.read_csv('./plots_for_thesis/100k_LayerNorm_Celux2_Linear_end_3000_epochs_All_Losses_Errors_batch_128.csv')
df_200k_cnn = pd.read_csv('./plots_for_thesis/200k_LayerNorm_Celux2_Linear_end_3000_epochs_All_Losses_Errors_batch_128.csv')
df_300k_cnn = pd.read_csv('./plots_for_thesis/300k_LayerNorm_Celux2_Linear_end_3000_epochs_All_Losses_Errors_batch_128.csv')
df_400k_cnn = pd.read_csv('./plots_for_thesis/400k_LayerNorm_Celux2_Linear_end_3000_epochs_All_Losses_Errors_batch_128.csv')
df_500k_cnn = pd.read_csv('./plots_for_thesis/500k_LayerNorm_Celux2_Linear_end_3000_epochs_All_Losses_Errors_batch_128.csv')





#Loss
df_50k_loss_cnn = list(df_50k_cnn.iloc[:, 2])
df_100k_loss_cnn = list(df_100k_cnn.iloc[:, 2])
df_200k_loss_cnn = list(df_200k_cnn.iloc[:, 2])
df_300k_loss_cnn = list(df_300k_cnn.iloc[:, 2])
df_400k_loss_cnn = list(df_400k_cnn.iloc[:, 2])
df_500k_loss_cnn = list(df_500k_cnn.iloc[:, 2])

loss_list_cnn = []
loss_list_cnn = [df_50k_loss_cnn, df_100k_loss_cnn, df_200k_loss_cnn, df_300k_loss_cnn, df_400k_loss_cnn, df_500k_loss_cnn]
loss_list_cnn = list(map(float, chain.from_iterable(loss_list_cnn)))


print(
    loss_list_cnn
)




plotdata_loss_cnn = pd.DataFrame({
    "Loss":loss_list_cnn,},
index = ['50K', '100K',
         '200K',  '300K',
         '400K',  '500K']
)





#Plot of sum of spending for "Car_Insurance_" for both DE and AUT for "Cost" metric.
ax = plotdata_loss_cnn.plot.bar(color=['blue'])
ax.bar_label(ax.containers[0], fmt='%6f', size=8)
plt.xticks(rotation=0, horizontalalignment="center", size=9)
plt.rcParams['figure.figsize'] = [5,5]
plt.rcParams['figure.dpi'] = 300
plt.title("Averaged loss for different dataset sizes - CNN")
plt.legend(loc='best')
plt.xlabel("Training dataset size")
plt.ylabel("Loss")
plt.savefig(f'./final_plots_thesis/Loss_Averaged_per_dataset_size_CNN.pdf', dpi=300,  bbox_inches='tight')
plt.show()




#Relative error


df_50k_rel_error_cnn = df_50k_cnn.iloc[0, 11]
#print("Test: ", df_50k_cnn)
#print("Test: ", df_50k_cnn.dtypes)
df_100k_rel_error_cnn = df_100k_cnn.iloc[0, 11]
df_200k_rel_error_cnn = df_200k_cnn.iloc[0, 11]
df_300k_rel_error_cnn = df_300k_cnn.iloc[0, 11]
df_400k_rel_error_cnn = df_400k_cnn.iloc[0, 11]
df_500k_rel_error_cnn = df_500k_cnn.iloc[0, 11]





# The '[' and ']' cause problems what I try to convert from str to float hence I need to replace them with '' .
print("The initial type is: ", type(df_50k_rel_error_cnn))

df_50k_rel_error_cnn = float(df_50k_rel_error_cnn.replace('[', '').replace(']', ''))
print("The new type type is: ", type(df_50k_rel_error_cnn))

df_100k_rel_error_cnn = float(df_100k_rel_error_cnn.replace('[', '').replace(']', ''))
df_200k_rel_error_cnn = float(df_200k_rel_error_cnn.replace('[', '').replace(']', ''))
df_300k_rel_error_cnn = float(df_300k_rel_error_cnn.replace('[', '').replace(']', ''))
df_400k_rel_error_cnn = float(df_400k_rel_error_cnn.replace('[', '').replace(']', ''))
df_500k_rel_error_cnn = float(df_500k_rel_error_cnn.replace('[', '').replace(']', ''))



relative_error_cnn = []
relative_error_cnn = np.array([df_50k_rel_error_cnn, df_100k_rel_error_cnn, df_200k_rel_error_cnn, df_300k_rel_error_cnn, df_400k_rel_error_cnn, df_500k_rel_error_cnn])
# Iterate over list



print(
    "This is the list: ",relative_error_cnn,
    "The shape is: relative_error_cnn", np.shape(relative_error_cnn)
)


plotdata_rel_error_cnn = pd.DataFrame({
    "Relative error":relative_error_cnn ,},
index = ['50K', '100K',
         '200K',  '300K',
         '400K',  '500K']
)



ax = plotdata_rel_error_cnn.plot.bar(color=['blue'])
ax.bar_label(ax.containers[0], fmt='%6f', size=8)
plt.xticks(rotation=0, horizontalalignment="center", size=9)
plt.rcParams['figure.figsize'] = [5,5]
plt.rcParams['figure.dpi'] = 300
plt.title("Averaged Relative error for different dataset sizes - CNN")
plt.legend(loc='best')
plt.xlabel("Training dataset size")
plt.ylabel(r'$\mathrm{Relative \ error_{avg}}$ $\left[\frac{1}{n}\sum_{}^{} \frac{|target - predicted|}{target}\right]$')
plt.savefig(f'./final_plots_thesis/Relative_error_Averaged_per_dataset_size_CNN.pdf', dpi=300,  bbox_inches='tight')
plt.show()





#Absolute error


df_50k_abs_error_cnn = df_50k_cnn.iloc[0, 11]
#print("Test: ", df_50k_cnn)
#print("Test: ", df_50k_cnn.dtypes)
df_100k_abs_error_cnn = df_100k_cnn.iloc[0, 11]
df_200k_abs_error_cnn = df_200k_cnn.iloc[0, 11]
df_300k_abs_error_cnn = df_300k_cnn.iloc[0, 11]
df_400k_abs_error_cnn = df_400k_cnn.iloc[0, 11]
df_500k_abs_error_cnn = df_500k_cnn.iloc[0, 11]





# The '[' and ']' cause problems what I try to convert from str to float hence I need to replace them with '' .
print("The initial type is: ", type(df_50k_abs_error_cnn))

df_50k_abs_error_cnn = float(df_50k_abs_error_cnn.replace('[', '').replace(']', ''))
print("The new type type is: ", type(df_50k_abs_error_cnn))

df_100k_abs_error_cnn = float(df_100k_abs_error_cnn.replace('[', '').replace(']', ''))
df_200k_abs_error_cnn = float(df_200k_abs_error_cnn.replace('[', '').replace(']', ''))
df_300k_abs_error_cnn = float(df_300k_abs_error_cnn.replace('[', '').replace(']', ''))
df_400k_abs_error_cnn = float(df_400k_abs_error_cnn.replace('[', '').replace(']', ''))
df_500k_abs_error_cnn = float(df_500k_abs_error_cnn.replace('[', '').replace(']', ''))



absolute_error_cnn = []
absolute_error_cnn = np.array([df_50k_abs_error_cnn, df_100k_abs_error_cnn, df_200k_abs_error_cnn, df_300k_abs_error_cnn, df_400k_abs_error_cnn, df_500k_abs_error_cnn])
# Iterate over list



print(
    "This is the list: ",absolute_error_cnn,
    "The shape is: relative_error_cnn", np.shape(absolute_error_cnn)
)


plotdata_abs_error_cnn = pd.DataFrame({
    "Absolute error":absolute_error_cnn ,},
index = ['50K', '100K',
         '200K',  '300K',
         '400K',  '500K']
)



ax = plotdata_abs_error_cnn.plot.bar(color=['blue'])
ax.bar_label(ax.containers[0], fmt='%6f', size=8)
plt.xticks(rotation=0, horizontalalignment="center", size=9)
plt.rcParams['figure.figsize'] = [5,5]
plt.rcParams['figure.dpi'] = 300
plt.title("Averaged Absolute error for different dataset sizes - CNN")
plt.legend(loc='best')
plt.xlabel("Training dataset size")
plt.ylabel(r'$\mathrm{Absolute \ error_{avg}}$ $\left[\frac{1}{n}\sum_{}^{} |target - predicted| \right]$')
plt.savefig(f'./final_plots_thesis/Absolute_error_Averaged_per_dataset_size_CNN.pdf', dpi=300,  bbox_inches='tight')
plt.show()

