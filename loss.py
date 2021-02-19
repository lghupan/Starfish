## Load VGG model

from tensorflow.keras.applications.vgg16 import VGG16
vgg16 = VGG16(weights='imagenet', include_top=False)
for layer in vgg16.layers:
    layer.trainable = False
vggs = Model(inputs=vgg16.input, outputs=vgg16.get_layer("block2_conv2").output)
print(vggs(np.zeros((1,224,224,3))).shape )  # check if model works


## Load Classification model

car_model = load_model('./model_cars/model_0.9797.h5')
for layer in car_model.layers:
    layer.trainable = False
        
# Get labels of training and test for classification
from scipy.io import loadmat
cars_train_annos = loadmat('./Dataset/cars/devkit/cars_train_annos.mat')['annotations'][0]
cars_train_label = [c[4][0][0] - 1 for c in cars_train_annos]    # add offset 1 to make index start from 0
cars_classes = len(list(set(cars_train_label)))

train_list_label = [ cars_train_label[i-1] for i in [int(l.split('/')[-1].split('.')[0]) for l in train_list]] 
test_list_label = [ cars_train_label[i-1] for i in [int(l.split('/')[-1].split('.')[0]) for l in test_list]]
train_list_label_cat = tf.keras.utils.to_categorical(np.repeat(train_list_label, 25), cars_classes)
test_list_label_cat = tf.keras.utils.to_categorical(np.repeat(test_list_label, 25), cars_classes)


## following are assorted loss functions

def msssim_loss(y_true, y_pred):
    return tf.constant(1.0) - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 1.0))


def ssim_loss(y_true, y_pred):
    return tf.constant(1.0) - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def ssim_loss_flatten(y_true, y_pred):
    return ssim_loss(tf.reshape(y_true, [tf.shape(y_true)[0], 56, 72, 3]), y_pred)


def classification_loss_single(y_true, y_pred):
    loss = tf.keras.losses.categorical_crossentropy(car_model(y_true*2.0-1.0), car_model(y_pred*2.0-1.0))
    return tf.reduce_mean(loss, axis=-1)


def ssim_classification_loss_single(y_true, y_pred):
    ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    class_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(car_model(y_true*2.0-1.0), car_model(y_pred*2.0-1.0)))
    return ssim_loss + 1.1*class_loss


def input_ssim_classification_loss_single(x_true): 
    def loss_helper(y_true, y_pred):   # y_true is None
        #y_true = x_true
        return ssim_classification_loss_single(x_true, y_pred)
    return loss_helper


# hybrid loss need hybrid label
# np.concatenate((x_train[:1000].reshape(1000, 56*72*3), train_list_label_cat[:1000]), axis=-1)
def ssim_classification_loss(y_true, y_pred):
    x_train = y_true[:,:56*72*3]
    train_list_label_cat = y_true[:,56*72*3:]
    return 1.0*ssim_loss_flatten(x_train, y_pred) + 0.1*classification_loss(train_list_label_cat, y_pred)


def vgg16_loss(y_true, y_pred):
    return tf.keras.losses.MSE(vgg16(y_true*2.0-1.0), vgg16(y_pred*2.0-1.0))


def input_vgg16_loss(x_true):
    def loss_helper(y_true, y_pred):   # y_true is None
        #y_true = x_true
        return vgg16_loss(x_true, y_pred)
    return loss_helper


def vggs_loss(y_true, y_pred):
    return tf.keras.losses.MSE(vggs(y_true*2.0-1.0), vggs(y_pred*2.0-1.0))