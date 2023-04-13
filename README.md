# Double-UNet
A implementation in tensorflow of a UNet model for image segmentation, with two custom loss functions.

The UNet model is created using (create_unet.py)[Double-UNet/create_unet.py] function with input size of 32x32 and 1 channel for model1, and 3 channels for the model2. The optimizer used is Adam with a learning rate of 0.001.

![image](https://user-images.githubusercontent.com/98736513/231711822-897bd4e9-d19f-4a78-9ac3-abf1b03c5dcc.png)
