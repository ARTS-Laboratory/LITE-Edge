import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import resample

plt.close('all')
"""
preprocess datasets from benchtop experiments
"""

test_names = ["0-1","0-5","0-10","0-20"]
train_names = ["0-1","1-2","2-3","3-4","4-5","5-6","6-7","7-8","8-9","9-10",\
               "10-11","11-12","12-13","13-14","14-15","15-16","16-17","17-18",\
               "18-19","19-20","20-21","0-20"]

for i in range(len(test_names) + len(train_names)):
    #%%
    name =(test_names + train_names)[i]
    typ = "Test" if i < 4 else "Train"
    ref = np.loadtxt("./" + typ + " Sweep/"+name+"VI.lvm", skiprows=23, delimiter=',')
    pkg = np.loadtxt("./" + typ + " Sweep/"+name+"Pkg.CSV", delimiter=',')
    
    ref, ref_t = ref[:,1], ref[:,0]
    pkg, pkg_t = pkg[:,1], pkg[:,0]
    
    # subtract off means, negate pkg
    m_r = np.mean(ref)
    m_p = np.mean(pkg)
    pkg = m_p - pkg
    ref = ref - m_r
    
    # upsample pkg to ref sampling
    # alias ref to 200 Hz bandwidth
    upsize = ref.size
    ref1, ref_t1 = resample(ref, pkg.size, ref_t)
    ref1, ref_t1 = resample(ref1, upsize, ref_t1)
    pkg1, pkg_t1 = resample(pkg, upsize, pkg_t)
    
    # peaks
    peak_ref, _ = find_peaks(ref1, height=.2, width=2)
    peak_pkg, _ = find_peaks(pkg1, height=.2, width=2)
    peak_ref = peak_ref[0]
    peak_pkg = peak_pkg[0]
    
    # remove time offset
    delta_pkg = pkg_t1[peak_pkg]
    delta_ref = ref_t1[peak_ref]
    pkg1 = pkg1[peak_pkg:]
    pkg_t1 = ref_t1[:-peak_pkg] # use ref_t
    ref1 = ref1[peak_ref:]
    ref_t1 = ref_t1[:-peak_ref]
    
    # plt.figure()
    # plt.plot(ref_t1, ref1, label='reference')
    # plt.plot(pkg_t1, pkg1, label='package')
    # plt.plot(ref_t[0], ref[0], marker='.')
    # plt.plot(pkg_t[0], pkg[0], marker='.')
    # plt.legend()
    
    in_pkg = np.logical_and(pkg_t>delta_pkg+.1, pkg_t<180+delta_pkg)
    pkg2 = pkg[in_pkg]
    pkg_t2 = pkg_t[in_pkg]; pkg_t2=pkg_t2-pkg_t2[0]
    # just use interp; it's good enough
    ref2 = np.interp(pkg_t2, ref_t-delta_ref-.1, ref)
    
    plt.figure()
    plt.plot(pkg_t2, ref2, label='reference')
    plt.plot(pkg_t2, pkg2, label='package')
    plt.legend()
    
    pkg2 = pkg2.reshape(-1, 1)
    ref2 = ref2.reshape(-1, 1)
    pkg_t2 = pkg_t2.reshape(-1, 1)
    
    d = np.append(pkg2, ref2, axis=1)
    d = np.append(d, pkg_t2, axis=1)
    
    np.savetxt("./preprocessed/" + typ + "/" + name + ".csv", d, delimiter=',')

#%% assemble test and training datasets into one array
for i, name in enumerate(train_names):
    d = np.loadtxt("./preprocessed/Train/" + name + ".csv", delimiter=',')
    x = d[:,0]
    y = d[:,1]
    if(i == 0):
        X = np.expand_dims(x, 0)
        Y = np.expand_dims(y, 0)
    else:
        l = min(X.shape[1], x.size)
        X = np.append(X[:,:l], np.expand_dims(x[:l], 0), 0)
        Y = np.append(Y[:,:l], np.expand_dims(y[:l], 0), 0)
np.save("./preprocessed/Train/X_train.npy", X)
np.save("./preprocessed/Train/Y_train.npy", Y)

for i, name in enumerate(test_names):
    d = np.loadtxt("./preprocessed/Test/" + name + ".csv", delimiter=',')
    x = d[:,0]
    y = d[:,1]
    if(i == 0):
        X = np.expand_dims(x, 0)
        Y = np.expand_dims(y, 0)
    else:
        l = min(X.shape[1], x.size)
        X = np.append(X[:,:l], np.expand_dims(x[:l], 0), 0)
        Y = np.append(Y[:,:l], np.expand_dims(y[:l], 0), 0)
np.save("./preprocessed/Test/X_test.npy", X)
np.save("./preprocessed/Test/Y_test.npy", Y)