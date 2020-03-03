import numpy as np
import keras as k
import lycon
import pandas as pd

import argparse

from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet121
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam, SGD

models = {'vgg16': VGG16, 'resnet50': ResNet50, 'mobilenetv2': MobileNetV2, 'densenet121': DenseNet121}

parser = argparse.ArgumentParser(description="Train a Keras model on the KaoKore dataset")
parser.add_argument('--arch', type=str, choices=models.keys(), required=True)
parser.add_argument('--label', type=str, choices=['gender', 'status'], required=True)
parser.add_argument('--root', type=str, required=True)

parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam')
parser.add_argument('--lr-adjust-freq' , type=int, default=10, help='How many epochs per LR adjustment (*=0.1)')
# For Adam:
#   Should use lr = 0.001
# For SGD:
#   This lr is appropriate for ResNet and models with batch normalization,
#   but too high for AlexNet and VGG. Use 0.01 as the initial learning
#   rate for AlexNet or VGG.
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (L2 penalty)')
# TODO: adjust_lr_freq

args = parser.parse_args()

df = pd.read_csv(f'{args.root}/labels.csv')
image_dir = f'{args.root}/images_256/'

num_classes = 2 if args.label == 'gender' else 4
gen_to_cls = {'male': 0, 'female': 1} if args.label == 'gender' else {'noble': 0, 'warrior': 1, 'incarnation': 2, 'commoner': 3}

def lr_scheduler(epoch):
    lr = args.lr * (0.1**(epoch // args.lr_adjust_freq))
    return lr

class ModelSequence(k.utils.Sequence):
    def __init__(self, df, batch_size):
        self.x = df['image'].apply(lambda x: f'{image_dir}/{x}').values
        self.u = df[args.label].values
        self.batch_size = batch_size
        
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    
    def load(self, fn):
        img = lycon.load(fn)
        img = lycon.resize(img, args.image_size, args.image_size)
        if img is None:
            print('failed to load', fn)
            return None
        return (img.astype(np.float32) / 127.5) - 1

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        for fn, u in zip(self.x[idx * self.batch_size:(idx + 1) * self.batch_size], self.u[idx * self.batch_size:(idx + 1) * self.batch_size]):
            img = self.load(fn)
            if img is not None:
                batch_x.append(img)
                batch_y.append(u)
        return np.array(batch_x), np.array(batch_y)

seq_train = ModelSequence(df[df.set == 'train'], batch_size=args.batch_size)
seq_valid = ModelSequence(df[df.set == 'dev'], batch_size=args.batch_size)

def build_model():
    input_tensor = Input(shape=(args.image_size, args.image_size, 3))

    base_model = models[args.arch](
        include_top=False,
        input_tensor=input_tensor,
        # input_shape=(args.image_size, args.image_size, 3),
        classes=num_classes,
        pooling='avg')
    

    output_tensor = Dense(num_classes, activation='softmax')(base_model.output)
    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model


opt = {'adam': Adam, 'sgd': SGD}[args.optimizer]

model = build_model()
model.compile(optimizer=opt(lr=args.lr#, momentum=args.momentum
),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

model.summary()

model.fit_generator(seq_train, epochs=args.epochs, verbose=1, validation_data=seq_valid,
                    callbacks=[LearningRateScheduler(lr_scheduler)])

print('Dev set: ' , model.evaluate_generator(seq_valid, verbose=1))

seq_test = ModelSequence(df[df.set == 'test'], batch_size=args.batch_size)
print('Test set: ' , model.evaluate_generator(seq_test, verbose=1))
