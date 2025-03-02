import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation
import math
"""
animation portion created in python
"""
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
#%%create clean output signal
fps=30
vis_points=15
anim_len = 5
upsample=20


#%% generate signals
n = fps*anim_len*upsample
L = fps*anim_len
T1 = n/2 # first period
T2 = n/7 # second period
A1 = 1
A2 = .4

s1 = A1*np.cos(np.linspace(0, 2*math.pi*n/T1, num=n, endpoint=False))
s2 = A2*np.cos(np.linspace(0, 2*math.pi*n/T2, num=n, endpoint=False))

output_signal=s1+s2

T3 = n/50
T4 = n/75
A3=.1
A4=.05

s3 = A3*np.cos(np.linspace(0, 2*math.pi*n/T3, num=n, endpoint=False))
s4 = A4*np.cos(np.linspace(0, 2*math.pi*n/T4, num=n, endpoint=False))

input_signal = s1+s2+s3+s4

output_sampled = output_signal.reshape(-1, upsample)[:,0]
input_sampled = input_signal.reshape(-1, upsample)[:,0]

input_signal = np.tile(input_signal, 2)
input_sampled = np.tile(input_sampled, 2)
output_signal = np.tile(output_signal, 2)
output_sampled = np.tile(output_sampled, 2)


#%% make animation pngs

input_sample_points = np.linspace(0, vis_points, num=vis_points, endpoint=False)
input_signal_points = np.linspace(0, vis_points, num=vis_points*upsample, endpoint=False)
output_sample_points = np.linspace(5.5*vis_points, 6.5*vis_points, num=vis_points, endpoint=False)
output_signal_points = np.linspace(5.5*vis_points, 6.5*vis_points, num=vis_points*upsample, endpoint=False)


fig, ax = plt.subplots(1,1, figsize=(6, 2))
plt.axis('off')
ax.set_xlim(0, 6.5*vis_points)
ax.set_ylim(np.min(input_signal)-.2, np.max(input_signal)+.2)
l1, = ax.plot([],[], c=cc[0], marker='.', linewidth=0)
l2, = ax.plot([],[], c=cc[0], linewidth=1)
l3, = ax.plot([],[], c=cc[1], marker='.', linewidth=0)
l4, = ax.plot([],[], c=cc[1], linewidth=1)


def animate(i):
    i_sam = input_sampled[i:i+vis_points]
    i_sig = input_signal[i*upsample : (i+vis_points)*upsample]
    o_sam = output_sampled[i:i+vis_points]
    o_sig = output_signal[i*upsample : (i+vis_points)*upsample]
    
    l1.set_data(input_sample_points, i_sam)
    l2.set_data(input_signal_points, i_sig)
    l3.set_data(output_sample_points, o_sam)
    l4.set_data(output_signal_points, o_sig)

ani = mpl.animation.FuncAnimation(fig, animate, frames=fps*anim_len, interval=1/30*1000)
ani.save("signal animation.avi", writer='ffmpeg', fps=30, dpi=542/2)
