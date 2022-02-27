#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
 * LCD - Legged Contact Detection
 *
 * Copyright 2022-2023 Stylianos Piperakis and Michalis Maravgakis, Foundation for Research and Technology Hellas (FORTH)
 * License: BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Foundation for Research and Technology Hellas (FORTH)
 *	 nor the names of its contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
'''




from tensorflow.keras.models import Sequential, Model,load_model, save_model
from tensorflow.keras.layers import  Dense, Dropout, Flatten, MaxPooling1D
import tensorflow as tf
import tempfile
import os
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
# Hotfix function
def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False, dir=os.getcwd()) as fd:
            save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        os.unlink(fd.name)
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False, dir=os.getcwd()) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = load_model(fd.name)
        os.unlink(fd.name)
        self.__dict__ = model.__dict__


    cls = Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

class mlp():
    def __init__(self):
        self.init = False
        make_keras_picklable()


    def setArchitecture(self, input_dim):
        # self.model = Sequential()
        # # Input
        # self.model.add(Dense(320, activation='relu',input_shape=(input_dim,1)))
        # # First hidden
        # self.model.add(Dense(272, activation='relu'))
        # # Dropout
        # self.model.add(Dropout(0.3))
        # # Second hidden
        # self.model.add(Dense(160, activation='relu'))
        # # Max pooling
        # self.model.add(MaxPooling1D(pool_size=2, name="MaxPooling1D"))
        # self.model.add(Flatten())
        # self.model.add(Dense(304, activation='relu'))
        # self.model.add(Dense(2, activation='softmax'))
        # self.model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
        # self.init = True

        self.model = Sequential()
        # Input
        self.model.add(Dense(32, activation='relu',use_bias = True,input_shape=(input_dim,1)))
        # First hidden
        self.model.add(Dense(32, activation='relu'))
        # Dropout
        self.model.add(Dropout(0.3))
        # Second hidden
        self.model.add(Dense(16, activation='relu'))
        # Max pooling
        self.model.add(MaxPooling1D(pool_size=2, name="MaxPooling1D"))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
        self.init = True


    def fit(self, x_train, y_train,  epochs_, batch_size_):
        self.model_log = self.model.fit(x_train, y_train,validation_split = 0.2, epochs=epochs_, batch_size=batch_size_,  verbose=1, shuffle=True)
