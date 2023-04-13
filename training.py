import tensorflow as tf

# Parameters
optimizer = tf.keras.optimizers.Adam(0.001) # Define the optimizer
num_epochs = 450 #  the number of epochs
multiplier = 100 # the value of the multiplier because the values of images are very small, so to have a better training I multiply all images by the multiplier
dim = 32 # the shape of the image
num_images = 113 # the number of images

# Define two UNet models
model1 = create_unet(input_size=(dim, dim, 1), classes=2)  
model2 = create_unet(input_size=(dim, dim, 3), classes=1)

# Define the error function for the first training step
def custom_loss1(y_true, y_pred):
    loss = K.square(y_pred - y_true)
    loss = loss * [800,200]
    loss = tf.reduce_sum(loss)/(dim*dim*num_images*multiplier)
    return loss

# Define the error function for the second training step
def custom_loss2(y_true, y_pred):
    loss = K.square(y_pred - y_true)
    loss = tf.reduce_sum(loss)/(dim*dim*num_images*multiplier)
    return loss

inputs = X_train*multiplier
targets1 = y_train12*multiplier
targets2 = y_train3*multiplier

losses1 = []
losses2 = []

# Define the training loop to perform both steps at once
for epoch in range(num_epochs):
    # First step : train the model with custom_loss1
    with tf.GradientTape() as tape1:
        # get prediction for the first step
        outputs1 = model1(inputs, training=True)
        # Calculate the loss for the first step
        loss1 = custom_loss1(targets1, outputs1)
    
    # Calculate gradients for the first step
    grads1 = tape1.gradient(loss1, model1.trainable_variables)
    # Update model weights for the first step
    optimizer.apply_gradients(zip(grads1, model1.trainable_variables))
    
    # Second step: train the model with custom_loss2
    with tf.GradientTape() as tape2:
        # Get predictions for the second step using the outputs from the first step
        outputs2 = model2(np.concatenate([inputs, outputs1], axis=3), training=True)
        # Calculate the loss for the second step
        loss2 = custom_loss2(targets2, outputs2)

    # Calculate gradients for the second step
    grads2 = tape2.gradient(loss2, model2.trainable_variables)
    # Update model weights for the second step
    optimizer.apply_gradients(zip(grads2, model2.trainable_variables))
    losses1.append(loss1)
    losses2.append(loss2)
    
print("Training finish")
