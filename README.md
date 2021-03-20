# Build-Your-Own-ResNet
Implement your own state of the art ResNet architectures with a simple to use keras-based High-Level API 

# Dependencies
```pip install tensorflow```

# How to use
1. Download the ResNet.py file and move it into working directory

```python
import ResNet
import tensorflow as tf

resnet = ResNet.Resnet(input_shape=(28, 28, 3))
resnet.add([
  tf.keras.layers.Conv2D(16, (1, 1), padding='valid'),
  # Similarly add as many layers as needed
])

resnet.add_identity_block([
  # List all layers to be connected by a shortcut.
])

resnet.add([
  tf.keras.layers.Dense(2, activation='softmax')
])

#above steps can be repeated endlessly to add as many layers and identity blocks as required

#build the model. You now have a keras model you can use as you please
model = resnet.build_model()

# compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#fit model
history = model.fit(X, y, epochs=100, batch_size=32)
```
