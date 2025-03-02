import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import resample
import math

plt.close('all')
"""
Only do necessary processing for the re-interpretted testing dataset
"""
def signaltonoise(sig, noisy_signal, dB=True):
    noise = sig - noisy_signal
    a_sig = math.sqrt(np.mean(np.square(sig)))
    a_noise = math.sqrt(np.mean(np.square(noise)))
    snr = (a_sig/a_noise)**2
    if(not dB):
        return snr
    return 10*math.log(snr, 10)

#%%
name ="0-10"
typ = "Test"

ref = np.loadtxt("./" + typ + " Sweep/"+name+"VI.csv", delimiter=',')
pkg = np.loadtxt("./" + typ + " Sweep/"+name+"Pkg.csv", delimiter=',')
#%%
ref, ref_t = ref[:,1], ref[:,0]
pkg, pkg_t = pkg[:,1], pkg[:,0]

plt.figure()
plt.plot(ref, label='reference')
plt.plot(pkg, label='package')
plt.legend()
plt.tight_layout()

ref = ref[605:]
pkg = pkg[605:]


x = np.expand_dims(pkg, 0)
y = np.expand_dims(ref, 0)

np.save('./preprocessed/Test/X_test.npy', x)
np.save('./preprocessed/Test/Y_test.npy', y)

# subtract off means, negate pkg
# m_r = np.mean(ref)
# m_p = np.mean(pkg)
# pkg = pkg - m_p
# ref = ref - m_r

# # upsample so that peaks can be aligned better
# upsize = pkg.size * 20
# ref1, ref_t1 = resample(ref, upsize, ref_t)
# pkg1, pkg_t1 = resample(pkg, upsize, pkg_t)

# # peaks
# peak_ref, _ = find_peaks(ref1, height=.2, width=2)
# peak_pkg, _ = find_peaks(pkg1, height=.2, width=2)
# peak_ref = peak_ref[0]
# peak_pkg = peak_pkg[0]

# # remove time offset
# delta_pkg = pkg_t1[peak_pkg]
# delta_ref = ref_t1[peak_ref]

# in_pkg = np.logical_and(pkg_t>delta_pkg+.1, pkg_t<180+delta_pkg)
# pkg = pkg[in_pkg]
# pkg_t = pkg_t[in_pkg]; pkg_t=pkg_t-pkg_t[0]
# ref = np.interp(pkg_t, ref_t1-delta_ref-.1, ref1)

# plt.figure()
# plt.plot(pkg_t, ref, label='reference')
# plt.plot(pkg_t, pkg, label='package')
# plt.legend()

# pkg = pkg.reshape(-1, 1)
# ref = ref.reshape(-1, 1)
# pkg_t = pkg_t.reshape(-1, 1)

# d = np.append(pkg, ref, axis=1)
# d = np.append(d, pkg_t, axis=1)

# np.savetxt("./preprocessed/" + typ + "/" + name + ".csv", d, delimiter=',')
# #%% assemble test and training datasets into one array
# for i, name in enumerate(train_names):
#     d = np.loadtxt("./preprocessed/Train/" + name + ".csv", delimiter=',')
#     x = d[:,0]
#     y = d[:,1]
#     if(i == 0):
#         X = np.expand_dims(x, 0)
#         Y = np.expand_dims(y, 0)
#     else:
#         l = min(X.shape[1], x.size)
#         X = np.append(X[:,:l], np.expand_dims(x[:l], 0), 0)
#         Y = np.append(Y[:,:l], np.expand_dims(y[:l], 0), 0)
# np.save("./preprocessed/Train/X_train.npy", X)
# np.save("./preprocessed/Train/Y_train.npy", Y)

# for i, name in enumerate(test_names):
#     d = np.loadtxt("./preprocessed/Test/" + name + ".csv", delimiter=',')
#     x = d[:,0]
#     y = d[:,1]
#     if(i == 0):
#         X = np.expand_dims(x, 0)
#         Y = np.expand_dims(y, 0)
#     else:
#         l = min(X.shape[1], x.size)
#         X = np.append(X[:,:l], np.expand_dims(x[:l], 0), 0)
#         Y = np.append(Y[:,:l], np.expand_dims(y[:l], 0), 0)
# np.save("./preprocessed/Test/X_test.npy", X)
# np.save("./preprocessed/Test/Y_test.npy", Y)
#%% produce phase shift in 0-10 Hz training data
# X_train = np.load("./preprocessed/Train/X_train.npy")
# Y_train = np.load("./preprocessed/Train/Y_train.npy")

# pkg = X_train[-1]
# ref = Y_train[-1]

# m = 1

# pkg = pkg[:-m]
# ref = ref[m:]

# print(signaltonoise(ref, pkg))

# plt.plot(pkg)
# plt.plot(ref)

# X_train = X_train[:,:-m]
# Y_train = Y_train[:,m:]

# np.save("./preprocessed/Train/X_train.npy", X_train)
# np.save("./preprocessed/Train/Y_train.npy", Y_train)