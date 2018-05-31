import os
import sys
sys.path.append('common')
import util, audio_preprocessing

import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser

class SingleDataset:
    '''
    - Train data flow
    [random erasor] -> [ImageDataGenerator] -> [MixupGenerator] ---> feed to fit()
    - Test data flow
    [ImageDataGenerator] ---> feed to predict()
    '''
    def __init__(self, prefix, labels, batch_size):
        self.name = prefix + 'dataset'
        # Load preprocessed data
        self.X_train = np.load(prefix+'X_train.npy')
        self.y_train = keras.utils.to_categorical(np.load(prefix+'y_train.npy'))
        self.X_valid = np.load(prefix+'X_valid.npy')
        self.y_valid = keras.utils.to_categorical(np.load(prefix+'y_valid.npy'))
        self.X_test = np.load(prefix+'X_test.npy')
        self.y_test = keras.utils.to_categorical(np.load(prefix+'y_test.npy'))
        # Make label from/to class converter
        self.labels = labels
        self.label2int = {l:i for i, l in enumerate(labels)}
        self.int2label = {i:l for i, l in enumerate(labels)}
        self.num_classes = len(self.labels)
        # Normalize
        max_amplitude = np.max(np.abs(np.vstack([self.X_train, self.X_valid, self.X_test])))
        self.X_train = self.X_train / max_amplitude
        self.X_valid = self.X_valid / max_amplitude
        self.X_test = self.X_test / max_amplitude
        # Add dimension [:, features, timesetep] -> [:, features, timestep, 1]
        self.X_train = self.X_train[..., np.newaxis]
        self.X_valid = self.X_valid[..., np.newaxis]
        self.X_test = self.X_test[..., np.newaxis]
        # Make data generators
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=0,
            width_shift_range=0.4,
            height_shift_range=0,
            horizontal_flip=True,
            preprocessing_function=get_random_eraser(v_l=np.min(self.X_train), v_h=np.max(self.X_train)) # RANDOM ERASER
        )
        datagen.fit(np.r_[self.X_train, self.X_valid, self.X_test])
        test_datagen = ImageDataGenerator(
            featurewise_center=datagen.featurewise_center,
            featurewise_std_normalization=datagen.featurewise_std_normalization
        )
        test_datagen.mean, test_datagen.std = datagen.mean, datagen.std
        self.datagen = datagen
        self.test_datagen = test_datagen
        self.batch_size = batch_size
        self.reset_generators()
    def reset_generators(self):
        mixupgen = MixupGenerator(self.X_train, self.y_train, alpha=1.0, batch_size=self.batch_size, datagen=self.datagen)
        self.train_generator = mixupgen() # datagen.flow(self.X_train, self.y_train)
        self.valid_generator = self.test_datagen.flow(self.X_valid, self.y_valid, batch_size=1, shuffle=False)
        self.test_generator = self.test_datagen.flow(self.X_test, self.y_test, batch_size=1, shuffle=False)
        #self.train_generator.reset() - no reset for mixupgen
        self.valid_generator.reset()
        self.test_generator.reset()
    def sizeof_trainset(self):
        return self.X_train.shape[0]
    def sizeof_validset(self):
        return self.X_valid.shape[0]
    def sizeof_testset(self):
        return self.X_test.shape[0]

class MultiDataset:
    """mix single datasets' X into one X set for specified mix ratio
    - Mixed data dimensions
    training data [n_train][n_feature][duration][1]
    validation data [n_validation][n_feature][duration][1]
    test data [n_test][n_feature][duration][1]
    """ 
    def __init__(self, single_datasets, x_mix_ratio, mix_randomness=0.2):
        self.single_datasets = single_datasets
        self.x_mix_ratio = x_mix_ratio
        self.mix_randomness = mix_randomness
        self.batch_size = single_datasets[0].batch_size
        self.n_feature = single_datasets[0].X_train.shape[1]
        self.duration = np.min([d.X_train.shape[2] for d in single_datasets]) # Shortest => mixed duration
        self.n_train = np.max([d.X_train.shape[0] for d in single_datasets]) # Largest => mixed number of data
        self.n_valid = np.max([d.X_valid.shape[0] for d in single_datasets]) # Largest => mixed number of data
        self.n_test = np.max([d.X_test.shape[0] for d in single_datasets]) # Largest => mixed number of data
        self.train_steps_per_epoch = self.n_train // self.batch_size
        self.valid_steps_per_epoch = sum([single.sizeof_validset() for single in single_datasets])
        self.test_steps_per_epoch = sum([single.sizeof_testset() for single in single_datasets])
        # Assertion
        assert util.all_elements_are_identical([d.X_train.shape[1] for d in single_datasets]), '1st dimension shall be identical.' 
        assert util.all_elements_are_identical([d.batch_size for d in single_datasets]), 'Batch size shall be identical.'
        self.reset()
    def reset(self):
        for single in self.single_datasets:
            single.reset_generators()
        self.train_generator = self._train_generator()
        self.valid_generator = self._vaild_generator()
        self.test_generator = self._test_generator()
    def autofill_batch(self, Xys, num_of_data):
        if util.all_elements_are_identical([num_of_data] + [len(Xy[0]) for Xy in Xys]):
            return Xys # for making faster
        # fix
        ensured = []
        for Xy in Xys:
            if len(Xy[0]) < num_of_data:
                fixed_X = np.pad(Xy[0], ((0, num_of_data - len(Xy[0])), (0,0), (0,0), (0,0)), 'reflect')
                fixed_y = np.pad(Xy[1], ((0, num_of_data - len(Xy[1])), (0,0)), 'reflect')
                print(Xy[0].shape, Xy[1].shape, 'to', fixed_X.shape, fixed_y.shape)
                Xy = (fixed_X, fixed_y)
            ensured.append(Xy)
        return ensured
    def _mixer(self, Xys, mix_ratio, batch_size):
        # Autofill to keep batch data size consistent among datasets
        Xys = self.autofill_batch(Xys, batch_size)
        # Randomize mix ratio
        k = np.random.uniform(low=-self.mix_randomness, high=self.mix_randomness)
        mix_ratio = [m + k for m in mix_ratio]
        # Divide to Xs and ys with mix ratio
        Xs = [Xy[0] * one_ratio for Xy, one_ratio in zip(Xys, mix_ratio)]
        ys = [Xy[1] * one_ratio for Xy, one_ratio in zip(Xys, mix_ratio)]
        # Mix Xs
        Xmix = np.zeros((batch_size, self.n_feature, self.duration, 1))
        for X in Xs:
            Xtemp = []
            for x_one in X:
                x_normed = util.random_unify_3d_mels(x_one, self.duration)
                Xtemp.append(x_normed)
            Xmix = Xmix + Xtemp
        return Xmix, ys
    def _train_generator(self):
        while True:
            yield self._mixer([next(single.train_generator) for single in self.single_datasets], self.x_mix_ratio, self.batch_size)
    def _vaild_generator(self):
        while True:
            yield self._mixer([next(single.valid_generator) for single in self.single_datasets], self.x_mix_ratio, 1)
    def _test_generator(self):
        while True:
            yield self._mixer([next(single.test_generator) for single in self.single_datasets], self.x_mix_ratio, 1)
    def train_single_generator(self):
        while True:
            Xys = self._mixer([next(single.train_generator) for single in self.single_datasets], self.x_mix_ratio, self.batch_size)
            yield [Xys[0], Xys[1][0]]
    def vaild_single_generator(self):
        while True:
            Xys = self._mixer([next(single.valid_generator) for single in self.single_datasets], self.x_mix_ratio, 1)
            yield [Xys[0], Xys[1][0]]
    def input_shape(self):
        return (self.n_feature, self.duration, 1)
    def ys_classes(self):
        return [single.num_classes for single in self.single_datasets]
    def evaluate_by_datasets(self, model):
        from keras.utils.generic_utils import Progbar
        results = []
        for i, single in enumerate(self.single_datasets):
            ys = [np.zeros(s.y_valid.shape[1:]) for s in self.single_datasets] # makes blank ys
            result = []
            print('Evaluating', single.name)
            progbar = Progbar(len(single.X_valid))
            for j in range(len(single.X_valid)):
                X, y = next(single.valid_generator)
                Xtemp = []
                for x_one in X:
                    x_normed = util.random_unify_3d_mels(x_one, self.duration)
                    Xtemp.append(x_normed)
                Xtemp = np.array(Xtemp)
                result.append(np.argmax(y) == np.argmax(model.predict(Xtemp)[i]))
                progbar.update(j)
            results.append(result)
            progbar.update(len(single.X_valid))
            print(' =', np.sum(result)/len(result))
        accuracies = [np.sum(result)/len(result) for result in results]
        for s, acc in zip(self.single_datasets, accuracies):
            print('Accuracy with %s = %f' % (s.name, acc))
        return accuracies
