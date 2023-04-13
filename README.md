# UNet with two custom losses
A implementation in tensorflow of a UNet model for image segmentation, with two custom loss functions. The purpose is to predict three images.

![image](https://user-images.githubusercontent.com/98736513/231716852-0fee01a3-f59a-461f-bcaf-86fd34c87a6c.png)


## Details

![image](https://user-images.githubusercontent.com/98736513/231711822-897bd4e9-d19f-4a78-9ac3-abf1b03c5dcc.png)

The UNet model is created using (create_unet.py)[Double-UNet/create_unet.py] function with input size of 32x32 and 1 channel for model1, and 3 channels for the model2. The optimizer used is Adam with a learning rate of 0.001.

The training loop in (training.py)[Double-UNet/training.py] is executed for a specified number of epoches and for each epoch, the model is trained with 'custom_loss1' and 'custom_loss2' in two steps. The losses are claculated using the 'tf.Gradient.Tape()' method. The gradient are then calculated and used to updtae the model weights using optimizer.

## Results 
![image](https://user-images.githubusercontent.com/98736513/231716006-9ffba13b-45de-4da2-9acb-7cada2889f05.png)


![image](https://user-images.githubusercontent.com/98736513/231715755-3377c3cb-7d9f-4e92-a32c-ce4e1125c92c.png)
