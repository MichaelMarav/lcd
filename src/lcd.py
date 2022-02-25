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

from sklearn import mixture
from mlp import mlp
import pickle as pickle
from tensorflow.keras.models import load_model
import os


class lcd():
    def __init__(self):
        self.init = False
	
    def setConfiguration(self,  robot_, humanoid_, load_model_ = False, out_path = "."):
        self.robot = robot_
        self.humanoid = humanoid_
        if(self.humanoid):
            self.input_dim = 12  #F/T + IMU
        else: 
            self.input_dim = 7   #Force + IMU

        if(load_model_):
            self.mlp = mlp()
            self.mlp.setArchitecture(self.input_dim)
            self.mlp = load_model(out_path  +'/'+self.robot + '_MLP',compile=False)
            self.model_log =  pickle.load(open(out_path  +'/'+self.robot + '_MLP_LOG', 'rb'))
            #self.gmm = pickle.load(open(out_path+'/'+self.robot + '_gmm.sav', 'rb'))
            self.init = True
        else:
            self.mlp = mlp()
            self.mlp.setArchitecture(self.input_dim)
            #self.gmm = mixture.GaussianMixture(n_components=3, covariance_type='full', max_iter=200, tol=5.0e-4, init_params = 'kmeans',warm_start=False, n_init=50, verbose=1)
          

    def fit(self,data_train,  data_labels,  epochs_, batch_size_, save_model_ = False):
        print("Data Size ",data_train.size)
        self.mlp.fit(data_train,data_labels, epochs_, batch_size_)
        self.leg_probabilities =  self.mlp.model.predict(data_train)
        self.model_log = self.mlp.model_log.history

        if(save_model_):
            self.mlp.model.save(self.robot + '_MLP')
            with open(self.robot + '_MLP_LOG', 'wb') as file_pi:
                pickle.dump(self.mlp.model_log, file_pi)
        #print("Clustering with Gaussian Mixture Models")
        #self.clusterGMM(save_model_)
        self.init = True


    def predict(self, data_):
        leg_probabilities = self.mlp.model.predict(data_.reshape(1,-1))
        #gait_phase = self.gmm.predict(leg_probabilities)
        #gait_phase_proba = self.gmm.predict_proba(leg_probabilities)
        return leg_probabilities #gait_phase, gait_phase_proba, 


    def predict_dataset(self, data_):
        leg_probabilities = self.mlp.model.predict(data_)
        #gait_phase = self.gmm.predict(leg_probabilities)

        return leg_probabilities #,gait_phase

  
    def clusterGMM(self, save_model_):
        self.gmm.fit(self.leg_probabilities)
        if(save_model_):
            with open(self.robot + '_gmm.sav', 'wb') as file:
                pickle.dump(self.gmm, file)
        self.gait_phases = self.gmm.predict(self.leg_probabilities)
