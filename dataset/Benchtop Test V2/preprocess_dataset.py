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
    
    ref = np.loadtxt("./" + typ + " Sweep/"+name+"VI.csv", delimiter=',')
    pkg = np.loadtxt("./" + typ + " Sweep/"+name+"Pkg.csv", delimiter=',')
    
    ref, ref_t = ref[:,1], ref[:,0]
    pkg, pkg_t = pkg[:,1], pkg[:,0]
    
    # subtract off means, negate pkg
    m_r = np.mean(ref)
    m_p = np.mean(pkg)
    pkg = pkg - m_p
    ref = ref - m_r
    
    # upsample so that peaks can be aligned better
    upsize = pkg.size * 20
    ref1, ref_t1 = resample(ref, upsize, ref_t)
    pkg1, pkg_t1 = resample(pkg, upsize, pkg_t)
    
    # peaks
    peak_ref, _ = find_peaks(ref1, height=.2, width=2)
    peak_pkg, _ = find_peaks(pkg1, height=.2, width=2)
    peak_ref = peak_ref[0]
    peak_pkg = peak_pkg[0]
    
    # remove time offset
    delta_pkg = pkg_t1[peak_pkg]
    delta_ref = ref_t1[peak_ref]
    
    in_pkg = np.logical_and(pkg_t>delta_pkg+.1, pkg_t<180+delta_pkg)
    pkg = pkg[in_pkg]
    pkg_t = pkg_t[in_pkg]; pkg_t=pkg_t-pkg_t[0]
    ref = np.interp(pkg_t, ref_t1-delta_ref-.1, ref1)
    
    plt.figure()
    plt.plot(pkg_t, ref, label='reference')
    plt.plot(pkg_t, pkg, label='package')
    plt.legend()
    
    pkg = pkg.reshape(-1, 1)
    ref = ref.reshape(-1, 1)
    pkg_t = pkg_t.reshape(-1, 1)
    
    d = np.append(pkg, ref, axis=1)
    d = np.append(d, pkg_t, axis=1)
    
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
