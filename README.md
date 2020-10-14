# Image Super Resolution  

## Examples
Below are some examples showing how to run the <code>main.py</code> demos. 

+ **EDSR** CVPR2017

<code>$ >python main.py --phase 'train' --hr_train_path 'DIV2K_train_HR/' --lr_train_path 'DIV2K_train_LR_x8/' --hr_val_path 
 'DIV2K_valid_HR/' --lr_val_path 'DIV2K_valid_LR_x8/' --scale 8 --res_scale 0.1 --last_act 'sigmoid' --normalization 1 --augment</code>

<code>$ >python main.py --phase 'test' --test_file 'images/0801x8.png' --scale 8 --last_act 'sigmoid' normalization 1</code>

|  LR   | HR | EDSR_sigmoid | EDSR_None|
|  ---- |  ---- | ----  | ----  |
| <img src="images/0801x8.png" /> | <img src="images/0801.png" /> | <img src="logs/result/EDSR/0801x8_sigmoid.png" /> | <img src="logs/result/EDSR/0801x8_None.png" /> |
| <img src="images/0829x8.png" /> | <img src="images/0829.png" /> | <img src="logs/result/EDSR/0829x8_sigmoid.png" /> | <img src="logs/result/EDSR/0829x8_None.png" /> |

+ **RCAN** ECCV2020

+ **DBPN** CVPR2018