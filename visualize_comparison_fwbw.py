"""
Visualization of NN training data - comparing forward and backward direction.

Malte Schilling, 11/14/2018
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

import pickle

###########################################
# Loading the training data and structure #
###########################################
# Structure of the Training data - different levels:
# 1) Top level list = for different architectures (size of hidden layer):
#     [0,1,2,4,8,16,32,64,128]
# 2) Next level list: multiple training runs from random initializations, n=10
# 3) Dict: contains 'loss', 'val_loss' as keys
# 4) and as entries on next level the associated time series (2000 learning iterations)
# Loading the training data from the pickle file
with open('Results/trainHistoryDict_5runs_5000ep_L1L2_backw', 'rb') as file_pi:
    hist_list_bw = pickle.load(file_pi) 
with open('Results/trainHistoryDict_5runs_5000ep_L1L2_forw', 'rb') as file_pi:
    hist_list_fw = pickle.load(file_pi) 
    
# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)] 
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.) 

    
# Construct data at end of training
copied_arch_val_loss = np.zeros((len(hist_list_fw), len(hist_list_fw[0])))
for arch_n in range(0,len(hist_list_fw)):
    for diff_runs in range(0, len(hist_list_fw[arch_n])):
        # Getting the last loss - at end of training
        copied_arch_val_loss[arch_n][diff_runs] = hist_list_fw[arch_n][diff_runs]['val_loss'][-1]
mean_arch_val_loss_fw = np.mean(copied_arch_val_loss, axis=1)
#print(copied_arch_val_loss[2])
std_arch_val_loss = np.std(copied_arch_val_loss, axis=1)
arch_val_loss_lower_std_fw = mean_arch_val_loss_fw - std_arch_val_loss
arch_val_loss_upper_std_fw = mean_arch_val_loss_fw + std_arch_val_loss 

copied_arch_val_loss = np.zeros((len(hist_list_bw), len(hist_list_bw[0])))
for arch_n in range(0,len(hist_list_bw)):
    for diff_runs in range(0, len(hist_list_bw[arch_n])):
        # Getting the last loss - at end of training
        copied_arch_val_loss[arch_n][diff_runs] = hist_list_bw[arch_n][diff_runs]['val_loss'][-1]
print("VAL LOSS: ", copied_arch_val_loss)
mean_arch_val_loss_bw = np.mean(copied_arch_val_loss, axis=1)
print(copied_arch_val_loss[2])
std_arch_val_loss = np.std(copied_arch_val_loss, axis=1)
arch_val_loss_lower_std_bw = mean_arch_val_loss_bw - std_arch_val_loss
arch_val_loss_upper_std_bw = mean_arch_val_loss_bw + std_arch_val_loss        


###########################################################
# 1 C - Comparison different Architectures after Training #
###########################################################
fig = plt.figure(figsize=(8, 6))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_arch = plt.subplot(111)  
ax_arch.spines["top"].set_visible(False)  
ax_arch.spines["right"].set_visible(False)  
     
ax_arch.set_yscale('log')
ax_arch.set_xlim(-1, 9)  
ax_arch.set_xticks(np.arange(0,len(hist_list_fw)))
ax_arch.set_xticklabels(['No Hidden L.','1','2','4','8','16','32','64','128'])

# Use matplotlib's fill_between() call to create error bars.    
plt.fill_between(range(0,len(mean_arch_val_loss_bw)), arch_val_loss_lower_std_bw,  
                 arch_val_loss_upper_std_bw, color=tableau20[1], alpha=0.5) 
plt.fill_between(range(0,len(mean_arch_val_loss_fw)), arch_val_loss_lower_std_fw,  
                 arch_val_loss_upper_std_fw, color=tableau20[3], alpha=0.5) 

plt.plot(range(0,len(mean_arch_val_loss_fw)), mean_arch_val_loss_bw, color=tableau20[0], lw=2)
plt.plot(range(0,len(mean_arch_val_loss_bw)), mean_arch_val_loss_fw, color=tableau20[2], lw=2)
plt.plot([-1,9], [60.95, 60.95], '--', color=tableau20[6], lw=2) #Squared error from Regression

print("MSE Backwards: ", mean_arch_val_loss_bw)
print("MSE Forward: ", mean_arch_val_loss_fw)

ax_arch.set_xlabel('# Hidden units', fontsize=14)
ax_arch.set_ylabel('MSE', fontsize=14)
#ax_arch.set_title('MSE after Learning', fontsize=20)   
py.savefig("Figures/MSE_Architecture_Comp.pdf")