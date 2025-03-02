# Low-rank Inference and Training Efficiency for Edge Computing (LITE-Edge)

## Train

1. open up visual studio code
1. pull up command prompt in vscode
1. in command prompt, execute the following. 
    ```shell
    cd script
    python train_lstm.py --data_root ../dataset --n_hist 1600 --batch_size 1024 --hidden_size 512 --n_iter 100 --lr 1e-5 --output_root ../output --n_workers 8
    ```
