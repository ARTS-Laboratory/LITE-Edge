import numpy as np
import matplotlib.pyplot as plt
import math
"""
Create random signal patterns for animation
"""
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
#%%create clean output signal
fps=30
vis_points=30
anim_len = 5



n = 400 # 100 point signal
T1 = n/2 # first period
T2 = n/7 # second period
A1 = 1
A2 = .4

s1 = A1*np.cos(np.linspace(0, 2*math.pi*n/T1, num=n, endpoint=False))
s2 = A2*np.cos(np.linspace(0, 2*math.pi*n/T2, num=n, endpoint=False))

output_signal=s1+s2

#%% create noisy input signal

T3 = n/50
T4 = n/75
A3=.1
A4=.05

s3 = A3*np.cos(np.linspace(0, 2*math.pi*n/T3, num=n, endpoint=False))
s4 = A4*np.cos(np.linspace(0, 2*math.pi*n/T4, num=n, endpoint=False))

input_signal = s1+s2+s3+s4

#%%
output_sampled = output_signal.reshape(20, -1)[:,0]
input_sampled = input_signal.reshape(20, -1)[:,0]

plt.figure(figsize=(8, 1))
plt.plot(output_signal, linewidth=2, c=cc[0])
plt.plot(np.arange(n, step=20), output_sampled, marker='o', linewidth=0, c=cc[0])
plt.axis('off')
plt.ylim([np.min(output_signal)-.2, np.max(output_signal)+.2])
# plt.tight_layout()
plt.savefig("output_signal.svg")

plt.figure(figsize=(8, 1))
plt.plot(input_signal, linewidth=2, c=cc[1])
plt.plot(np.arange(n, step=20), input_sampled, marker='o', linewidth=0, c=cc[1])
plt.axis('off')
plt.ylim([np.min(input_signal)-.2, np.max(input_signal)+.2])
# plt.tight_layout()
plt.savefig("input_signal.svg")

#%% make animation pngs
fps=30


i=0



output_

plt.figure(figsize=(6,3))
input_x_points = np.linspace(0, 30)



