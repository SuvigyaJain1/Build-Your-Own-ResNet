from './ResNet' import ResnetBase
from keras import activations, layers

model = ResnetBase((32,32,3), activations.relu)

model.add([
           layers.Conv2D(kernel_size=(7,7), strides=(1,1), filters=64, padding="same"),
           layers.BatchNormalization(),
           layers.MaxPooling2D(pool_size=(2,2))
           ], activation=tf.keras.activations.relu)

model.add_conv_bn_block(filters=[64, 64], strides=(1,1), kernel_sizes=[(3,3), (3,3)])
model.add_conv_bn_block(filters=[64, 64], strides=(1,1), kernel_sizes=[(3,3), (3,3)])
model.add_conv_bn_block(filters=[128, 128], strides=(1,1), kernel_sizes=[(3,3), (3,3)])
model.add_conv_bn_block(filters=[128, 128], strides=(1,1), kernel_sizes=[(3,3), (3,3)])
model.add_conv_bn_block(filters=[256, 256], strides=(2,2), kernel_sizes=[(3,3), (3,3)])
model.add_conv_bn_block(filters=[256, 256], strides=(1,1), kernel_sizes=[(3,3), (3,3)])
model.add_conv_bn_block(filters=[512, 512], strides=(2,2), kernel_sizes=[(3,3), (3,3)])
model.add_conv_bn_block(filters=[512, 512], strides=(1,1), kernel_sizes=[(3,3), (3,3)])


model.add([
           layers.AveragePooling2D(pool_size=(4,4)),
           layers.Flatten(),
           layers.Dense(10),
           layers.Softmax()
])

m = model.build_model()
m.summary()

m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
# history = m.fit(X_train, y_train, epochs=20)