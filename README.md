# Action Recognition on the KTH dataset

This work replicates the best experiment conducted by Aya Ismail et. al. as available on their [github repository: vkhoi/KTH-Action-Recognition](https://github.com/vkhoi/KTH-Action-Recognition). This work reproduces the results for recognising KTH Action classifications with CNN and Optical Flow, however the code is updated to use the [Pytorch Lightning framework](https://pytorch-lightning.rtfd.io/en/latest/) 


## KTH dataset
Official web page of KTH dataset: [link](http://www.nada.kth.se/cvap/actions). 
The KTH dataset consists of videos of humans performing 6 types of action: boxing, handclapping, handwaving, jogging, running, and walking. There are 25 subjects performing these actions in 4 scenarios: outdoor, outdoor with scale variation, outdoor with different clothes, and indoor. The total number of videos is therefore 25x4x6 = 600. The videos' frame rate are 25fps and their resolution is 160x120.

## Get Started
Running this model under Lightning means all of the Trainer command line arguements are available. Eg run the command line:

`python Train_Evaluate_KTH_VideoBlockClassifier.py --data_path <your data directory> --gpus 1 --max_epochs 200`
 
The Lightning framework also provides `--help` option to list all the available options for training.

## Results
Action recognition of the KTH dataset was attempted only with the following approaches:
* CNN on block of frames: 
* CNN on block of frames + optical flow: 

In above the accuracy of <mark>xx.xx%</mark>was achieved for the second experiemnet (compared to the reported 90.27% accuracy achieved by the original authors)

The lines of code was reduced by a factor of <mark>xxx</mark> with the following counts improved: 
<p/>
<table>
<thead><tr><th> Old filename </th><th> loc </th><th> New filename </th><th> loc </th></tr></thead>
<tbody>
<tr><td> dataset.py </td><td > 170 (51) </td><td rowspan=2> KTH_DataModule.py </td><td rowspan=2> 73 </td></tr>
<tr><td> data_utils.py  </td><td > 145 (83) </td></tr>
<tr><td> models/cnn_block_frame.py </td><td > 35 </td><td> KTH_VideoBlockClassifier.py </td><td> 55 </td></tr>
<tr><td> train_cnn_block_frame.py </td><td > 48 </td><td rowspan=3>Train_Evaluate_KTH_VideoBlockClassifier.py </td><td rowspan=3> 9 </td></tr>
<tr><td> train_helper.py  </td><td > 103 (94) </td></tr>
<tr><td> eval_cnn_block_frame.py  </td><td > 58 </td></tr>
</tbody>
<tfoot><tr><td><b>Total:</b></td><td><b>369</b></td><td>&nbsp;</td><td><b>137</b></td></tr></tfoot>
</table>


## References 
https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09