UNet model and other classic models based on UNet, such as UNet++,Attention_UNet, ResUNet, etc. This project is dedicated to realizing these classic networks in the same project, and realizing one-click switching between models, which is convenient for experimental comparison.

In this experiment, I added the most commonly used semantic segmentation evaluation indicators, such as dice coefficient, Hausdorff distance, average pixel accuracy and so on. At the same time, wandb is used to record these indexes and experimental losses.

First, you need to prepare your own data set, including training pictures and their corresponding annotations. We need to set the path of lines 27~32 in train.py, ```dir_img``` is used to store training pictures, ```dir_mask``` is used to store its corresponding annotations, and ```dir_checkpoint``` is used to store the model weights you have trained.


```
dir_img = r''
dir_mask = r''
dir_checkpoint = Path('')
```


If you want to train these Models from scratch, you only need to change line 265 in the train.py file and replace the name of the model with the name of the model you want to train. All the models available for training in this project are stored in the Models directory.

```model = Model(n_channels=3, n_classes=args.classes,bilinear = False)```

Of course, you also need to know whether the trained data set is single channel or three channels and modify the parameter ```n_channels```. At the same time, you also need to know what the final segmentation target is and modify the parameter ```n_classes```. ```Bilinear``` determines whether to perform linear interpolation when upsampling, and if not, transposed convolution will be performed.
