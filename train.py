import tensorflow as tf

# Définir votre modèle UNet avec custom_loss1 et custom_loss2
model1 = create_unet(input_size=(32,32,1), classes=2)  
model2 = create_unet(input_size=(32,32,3), classes=1)

# Définir votre optimiseur et les métriques que vous souhaitez utiliser pour suivre les performances de l'entraînement
optimizer = tf.keras.optimizers.Adam(0.001)
num_epochs = 450
a=100

# Définir la fonction d'erreur pour la première étape de l'entraînement
def custom_loss1(y_true, y_pred):
    loss = K.square(y_pred - y_true)
    loss = loss * [800,200]
    loss = tf.reduce_sum(loss)/(32*32*113*a)
    return loss

# Définir la fonction d'erreur pour la deuxième étape de l'entraînement
def custom_loss2(y_true, y_pred):
    loss = K.square(y_pred - y_true)
    loss = tf.reduce_sum(loss)/(32*32*113*a)
    return loss

inputs = X_train*a
targets1 = y_train12*a
targets2 = y_train3*a

mse = tf.keras.losses.MeanSquaredError()
losses1 = []
losses2 = []

# Définir votre boucle d'entraînement pour effectuer les deux étapes en une seule fois
for epoch in range(num_epochs):
    # Première étape: entraîner le modèle avec custom_loss1
    with tf.GradientTape() as tape1:
        # Obtenir les prédictions pour la première étape
        outputs1 = model1(inputs, training=True)
        # Calculer la perte pour la première étape
        # loss1 = mse(targets1, outputs1)
        loss1 = custom_loss1(targets1, outputs1)
    
    # Calculer les gradients pour la première étape
    grads1 = tape1.gradient(loss1, model1.trainable_variables)
    # Mettre à jour les poids du modèle pour la première étape
    optimizer.apply_gradients(zip(grads1, model1.trainable_variables))
    
    # Deuxième étape: entraîner le modèle avec custom_loss2
    with tf.GradientTape() as tape2:
        # Obtenir les prédictions pour la deuxième étape en utilisant les sorties de la première étape
        outputs2 = model2(np.concatenate([inputs, outputs1], axis=3), training=True)
        # Calculer la perte pour la deuxième étape
        # loss2 = mse(targets2, outputs2)
        loss2 = custom_loss2(targets2, outputs2)
    
    # Calculer les gradients pour la deuxième étape
    grads2 = tape2.gradient(loss2, model2.trainable_variables)
    # Mettre à jour les poids du modèle pour la deuxième étape
    optimizer.apply_gradients(zip(grads2, model2.trainable_variables))
    losses1.append(loss1)
    losses2.append(loss2)
    
print("Training finish")
