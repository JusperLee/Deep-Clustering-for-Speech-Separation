## Deep Clustering for Speech Separation
Deep clustering in the field of speech separation implemented by pytorch

Demo Pages: [Results of pure speech separation model](https://www.likai.show/Pure-Audio/index.html)

> Hershey J R, Chen Z, Le Roux J, et al. Deep clustering: Discriminative embeddings for segmentation and separation[C]//2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2016: 31-35.

## Requirement

- **Pytorch 1.3.0**
- **librosa 0.7.1**
- **PyYAML 5.1.2**


## Code writing log
**2019-12-27 Friday**. It is currently being refined and is not yet complete.

**2020-01-02 Thursday**. The training code is currently complete and the code bug is being tested.

## Training steps
1. First, you can use the create_scp script to generate training and test data scp files.

```shell
python create_scp.py
```

2. Then, in order to reduce the mismatch of training and test environments. Therefore, you need to run the util script to generate a feature normalization file (CMVN).

```shell
python ./utils/util.py
```

3. Finally, use the following command to train the network.

```shell
python train.py -opt ./option/train.yml
```

## Inference steps
1. Use the following command to start testing the model

```shell
python test.py -scp 1.scp -opt ./option/train.yml -save_file ./result
```
2. You can use the [this code](https://github.com/JusperLee/Calculate-SNR-SDR "this code") to calculate the SNR scores.


## Thanks

1. [Pytorch Template](https://github.com/victoresque/pytorch-template "Pytorch Template")
