import numpy as np

# Saves weight matrices to binary files
def export_binary(model, savpath = "./model_binaries"):
    import os
    import os.path as path
    from numpy import savetxt
    if(not path.exists(savpath)):
        os.mkdir(savpath)
    for layer in model.layers[:-1]: # the final layer is a dense top
        layer_path = savpath + "/" + "lstm" + "/"
        if(not path.exists(layer_path)):
            os.mkdir(layer_path)
        W, U, b = layer.get_weights()
        units = U.shape[0]
        
        # Concatonate U to W
        wi = np.hstack((W[:,:units].T,U[:,:units].T)).flatten()
        wf = np.hstack((W[:,units:units*2].T, U[:,units:units*2].T)).flatten()
        wc = np.hstack((W[:,units*2:units*3].T, U[:,units*2:units*3].T)).flatten()
        wo = np.hstack((W[:,units*3:].T, U[:,units*3:].T)).flatten()

        w = np.hstack((wi, wf, wc, wo))

        bi = b[:units].flatten()
        bf = b[units:units*2].flatten()
        bc = b[units*2:units*3].flatten()
        bo = b[units*3:].flatten()
        
        w.astype('<f4').tofile(layer_path + 'w.dat')

        bi.astype('<f4').tofile(layer_path + 'bI.dat')
        bf.astype('<f4').tofile(layer_path + 'bF.dat')
        bc.astype('<f4').tofile(layer_path + 'bC.dat')
        bo.astype('<f4').tofile(layer_path + 'bO.dat')

    #save dense top layer
    dense_top = model.layers[-1]
    in_weights, out_weights = dense_top.get_weights()
    layer_path = savpath + "/dense_top/"
    if(not path.exists(layer_path)):
        os.mkdir(layer_path)

    in_weights.astype('<f4').tofile(layer_path + 'w.dat')
    out_weights.astype('<f4').tofile(layer_path + 'b.dat')
