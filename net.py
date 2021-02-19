from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Reshape, Flatten, GlobalAveragePooling2D, Dropout


# https://tensorflow.github.io/compression/docs/entropy_bottleneck.html
# We’ve found that it generally helps not to use batch normalization, 
# and to sandwich the bottleneck between two linear transforms or convolutions 
# (i.e. to have no nonlinearities directly before and after).

# Only tf.compat.v1.image.resize_nearest_neighbor can be use for upscaling. 
# UpSampling2D and tf.image.resize cannot be converted
# tf.compat.v1.image.resize_nearest_neighbor(nn_in, [224, 224])
 
def build_enc_patch(enc_patch_in, C_size, pw, drop_out=0.0):
    #x = tf.pad(enc_patch_in, [[0,0],[pw, pw],[pw, pw],[0,0]], "CONSTANT")
    x = tf.keras.layers.ZeroPadding2D((pw,pw), name='pad')(enc_patch_in)
    x = Conv2D(C_size[0], (3, 3), strides=(1, 1), activation='relu', padding='same', name='econv1')(x)
    x = Conv2D(C_size[1], (3, 3), strides=(2, 2), activation='relu', padding='same', name='econv2')(x)
    x = Conv2D(C_size[2], (3, 3), strides=(2, 2), activation='relu', padding='same', name='econv3')(x)
    x = Conv2D(C_size[3], (3, 3), strides=(2, 2), activation=None, padding='same', name='econv4')(x)
    x = Flatten(name='flatten')(x)
    if drop_out>0.01:
        x = Dropout(0.4)(x)  # train network with dropout to simulate data loss
    embed_size = x.shape
    enc_patch = Model(enc_patch_in, x)
    return enc_patch, embed_size


def build_dec_patch(dec_patch_in, C_size, input_c, input_h, input_w, pw, use_upsampling2d=False):
    x = dec_patch_in
    #x = Dense(((input_h+pw*2)//8)*((input_w+pw*2)//8)*(C_min), activation=None, name='ddense1')(x)
    x = Reshape(((input_h+pw*2)//8, (input_w+pw*2)//8, input_c), name='reshape')(x)
    x = Conv2D(C_size[0], (3, 3), activation=None, padding='same', name='dconv1')(x)
    if use_upsampling2d:
        x = tf.compat.v1.keras.layers.UpSampling2D((2, 2), name='u0')(x)
    else:
        x = tf.compat.v1.image.resize_nearest_neighbor(x, [x.shape[1]*2, x.shape[2]*2])
    x = Conv2D(C_size[1], (3, 3), activation='relu', padding='same', name='dconv2')(x)
    if use_upsampling2d:
        x = tf.compat.v1.keras.layers.UpSampling2D((2, 2), name='u1')(x)
    else:
        x = tf.compat.v1.image.resize_nearest_neighbor(x, [x.shape[1]*2, x.shape[2]*2])
    x = Conv2D(C_size[2], (3, 3), activation='relu', padding='same', name='dconv3')(x)
    if use_upsampling2d:
        x = tf.compat.v1.keras.layers.UpSampling2D((2, 2), name='u2')(x)
    else:
        x = tf.compat.v1.image.resize_nearest_neighbor(x, [x.shape[1]*2, x.shape[2]*2])
    x = Conv2D(C_size[3], (3, 3), activation='relu', padding='same', name='dconv4')(x)
    x = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='dout')(x)
    #x = tf.slice(x, [0, pw, pw, 0], [-1, input_h, input_w, 3])
    x =tf.keras.layers.Cropping2D(((pw, pw)), name='crop')(x)
    dec_patch = Model(dec_patch_in, x)
    return dec_patch


class UniformNoise(Layer):
    # https://github.com/RitwikGupta/Keras-UniformNoise
    """Apply additive uniform noise
    Only active at training time since it is a regularization layer.
    # Arguments
        minval: Minimum value of the uniform distribution
        maxval: Maximum value of the uniform distribution
    # Input shape
        Arbitrary.
    # Output shape
        Same as the input shape.
    """

    def __init__(self, minval=-1.0, maxval=1.0, **kwargs):
        super(UniformNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.minval = minval
        self.maxval = maxval

    def call(self, inputs, training=None):
        def noised():
            return inputs + K.random_uniform(shape=K.shape(inputs),
                                             minval=self.minval,
                                             maxval=self.maxval)
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'minval': self.minval, 'maxval': self.maxval}
        base_config = super(UniformNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    
def save_q_model(model, fname, cal_dataset):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] 
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    #converter.allow_custom_ops = True

    ds = tf.data.Dataset.from_tensor_slices((cal_dataset)).batch(1)
    def representative_data_gen():
      for input_value in ds.take(10):
         yield [input_value]

    converter.representative_dataset = representative_data_gen

    tflite_model_quant = converter.convert()
    open (fname, "wb") .write(tflite_model_quant)
    print("Saved quantized tflite model " + fname)

    
def infer_q_tflite_single_local(fn, s):
    import tensorflow as tf
    interpreter = tf.compat.v1.lite.Interpreter(fn)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], s)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


def split_k210_model(m, enc_out, dec_in):
    enc_model = Model(inputs=m.input, outputs=m.get_layer(enc_out).output)
    dec_input = Input(enc_model.output.shape[1:])
    dec_model = dec_input
    dec_idx = list([l.name for l in m.layers]).index(dec_in)
    for l in m.layers[dec_idx:]:
        dec_model = l(dec_model)
    dec_model = Model(inputs=dec_input, outputs=dec_model)
    
    #print(enc_model.summary())
    #print(dec_model.summary())
    test_in = np.zeros([10] + m.input.shape[1:])
    
    print("Model difference:", np.sum(np.abs(m.predict(test_in) - dec_model(enc_model(test_in)))))
    
    return enc_model, dec_model