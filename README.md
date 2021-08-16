# Action Recognition on the KTH dataset

This work replicates the best experiment conducted by Aya Ismail et. al. as available on their [github repository: vkhoi/KTH-Action-Recognition](https://github.com/vkhoi/KTH-Action-Recognition). This work reproduces the results for recognising KTH Action classifications with CNN and Optical Flow, however the code is updated to use the [Pytorch Lightning framework](https://pytorch-lightning.rtfd.io/en/latest/) 


## KTH dataset
Official web page of KTH dataset: [link](http://www.nada.kth.se/cvap/actions). 
The KTH dataset consists of videos of humans performing 6 types of action: boxing, handclapping, handwaving, jogging, running, and walking. There are 25 subjects performing these actions in 4 scenarios: outdoor, outdoor with scale variation, outdoor with different clothes, and indoor. The total number of videos is therefore 25x4x6 = 600. The videos' frame rate are 25fps and their resolution is 160x120.

## Results
Action recognition of the KTH dataset was attempted only with the following approaches:
* CNN on block of frames: 
* CNN on block of frames + optical flow: 

In above the accuracy of <mark>xx.xx%</mark>was achieved for the second experiemnet (compared to the reported 90.27% accuracy achieved by the original authors)

The lines of code was reduced by a factor of <mark>xxx</mark> with the following counts improved: 
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| main/data_utils.py | Python | 145 | 29 | 44 | 218 |
| [main/dataset.py](/main/dataset.py) | Python | 170 | 1 | 52 | 223 |
| [main/eval_cnn_block_frame.py](/main/eval_cnn_block_frame.py) | Python | 58 | 2 | 19 | 79 |
| [main/eval_cnn_block_frame_flow.py](/main/eval_cnn_block_frame_flow.py) | Python | 82 | 2 | 28 | 112 |
| [main/eval_cnn_single_frame.py](/main/eval_cnn_single_frame.py) | Python | 52 | 2 | 18 | 72 |
| [main/models/cnn_block_frame.py](/main/models/cnn_block_frame.py) | Python | 35 | 4 | 10 | 49 |
| [main/models/cnn_block_frame_flow.py](/main/models/cnn_block_frame_flow.py) | Python | 84 | 26 | 12 | 122 |
| [main/models/cnn_single_frame.py](/main/models/cnn_single_frame.py) | Python | 35 | 4 | 10 | 49 |
| [main/train_cnn_block_frame.py](/main/train_cnn_block_frame.py) | Python | 48 | 2 | 11 | 61 |
| [main/train_cnn_block_frame_flow.py](/main/train_cnn_block_frame_flow.py) | Python | 49 | 2 | 11 | 62 |
| [main/train_cnn_single_frame.py](/main/train_cnn_single_frame.py) | Python | 48 | 2 | 11 | 61 |
| [main/train_helper.py](/main/train_helper.py) | Python | 103 | 11 | 42 | 156 |


## References 
https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09