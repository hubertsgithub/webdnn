import keras
from keras.layers import Input
from webdnn.frontend.keras import KerasConverter
from webdnn.backend import generate_descriptor

from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.inception_v3 import InceptionV3
from keras.applications.nasnet import NASNetMobile,NASNetLarge
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception


#models = [ResNet50, DenseNet121, DenseNet169, DenseNet201, InceptionV3, NASNetMobile, NASNetLarge, VGG19, Xception]
# models = [InceptionV3, VGG19, Xception]

model = keras.models.load_model('/home/hlin/Ubuntu16VM_shared_folder/projects/colorization_main_net.h5')

models = [model]
for modelf in models:
    print('>>>>> {}'.format(modelf))
    # model = modelf(include_top=True, weights=None)

    # data_l = Input(shape=(256,256,1))
    # data_ab_mask = Input(shape=(256,256,3))
    # model = modelf(inputs = [data_l, data_ab_mask])
    model = modelf

    # model.save("resnet50.h5")
    # model.summary()

    try:
        graph = KerasConverter(batch_size=1).convert(model)
    except Exception as e:
        print ('*********Failed on {}'.format(modelf))
        print (e)
    print('--------------------------------')

exit(1)
exec_info = generate_descriptor("webgl", graph)  # also "webassembly", "webgl", "fallback" are available.

output_dir = "./keras_resnet50_output"
print("Saving to {}".format(output_dir))
exec_info.save(output_dir)
