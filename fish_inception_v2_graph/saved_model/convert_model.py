import tensorflow as tf

loaded = tf.saved_model.load("/media/saleh/20A8BE67A8BE3ADC/University/project/Project/My project/fish_detection/fish_inception_v2_graph/saved_model/")


class LayerFromSavedModel(tf.keras.layers.Layer):
    def __init__(self):
        super(LayerFromSavedModel, self).__init__()
        self.vars = loaded.variables

    def call(self, inputs):
        return loaded.signatures['serving_default'](inputs)


input = tf.keras.Input(...)
model = tf.keras.Model(input, LayerFromSavedModel()(input))
model.save('converted_model')
