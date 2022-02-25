import numpy as np
import pandas as pd
import math
from numpy import mean,std
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import random
from lcd import lcd


def read_dataset(filename):
  df = pd.read_csv(filename,engine = 'python')
  dataset = df.values
  dataset = dataset.astype('float32')

  print("Total data -> ", dataset.shape[0])
  print("Slip data -> ",np.count_nonzero(dataset[:,-1]==2))
  print("Stable data -> ",np.count_nonzero(dataset[:,-1]==0))

  return dataset


def remove_fly_data(data):
  indx2remove = []
  for i in range(0,data.shape[0]):
    if data[i,-1] == 1:
      indx2remove.append(i)
    if data[i,-1] == 2:
      data[i,-1] = 1
  data = np.delete(data,indx2remove,axis = 0)
  return data

def add_noise(data,std):
  mu = 0  # mean and standard deviation
  s = np.random.normal(mu, std, data.shape[0])
  for f in range(0,data.shape[1]):
    for i in range(0,data.shape[0]):
      data[i,f] = data[i,f] + s[i]
  return data


# Normalize stelios
# def normalize(din, dmax):
#     if(dmax != 0):
#         dout =  np.abs(din/dmax)
#     else:
#         dout =  np.zeros((np.size(din)))
#     return dout



# Use this before split data and labels
def normalize(data):
  result = deepcopy(data)
  result = abs(result)
  maxes = []
  for f in range(data.shape[1]-1):
    maxes.append(np.max(result[:,f]))

  for f in range(data.shape[1]-1):
    # max_f = np.max(data[:,f])
    for i in range(data.shape[0]):
      data[i,f] = data[i,f]/maxes[f]

  return data


def standard(data):
  for f in range(data.shape[1]-1):
    mean_f = np.mean(data[:,f])
    std_f  = np.std(data[:,f])
    for i in range(data.shape[0]):
      data[i,f] = (data[i,f]-mean_f)/std_f
  return data

def remove_features(features_to_remove,dataset):
  dataset = np.delete(dataset,features_to_remove,axis=1)
  return dataset


def z_score_outlier_detection(data,threshold):
  samples_to_remove = []
  mean_f = [0]
  std_f  = [0]
  for f in range(1,data.shape[1]-1):
    mean_f.append(np.mean(data[:,f]))
    std_f.append(np.std(data[:,f]))

  for i in range(data.shape[0]):
    for f in range(1,data.shape[1]-1):
      z = abs(mean_f[f] - dataset[i,f])/std_f[f]
      if z > threshold:
        samples_to_remove.append(i)
        break

  print("REMOVED SAMPLES : ",len(samples_to_remove))

  data = np.delete(data,samples_to_remove,axis = 0)

  return data

# remove outliers based on the mass of the robot

def remove_outliers(data,mass,pc_cut):
    g = 9.81
    samples_to_remove = []
    for i in range(data.shape[0]):
        if data[i,0] > mass*g*(1+pc_cut):  # remove all data that are greater that (100%+oc_cut%) of mg
            samples_to_remove.append(i)

    data = np.delete(data,samples_to_remove,axis=0)
    return data


if __name__ == "__main__":
    dataset = read_dataset('ATLAS_21k_02ground_coul_vel.csv')

    dataset = remove_fly_data(dataset)

    dataset = remove_features([0,1,3,4,5],dataset)

    dataset[:,0:1] = add_noise(dataset[:,0:1],0.6325)       # Fz
    dataset[:,1:4] = add_noise(dataset[:,1:4],0.0078)       # ax ay az
    dataset[:,4:7] = add_noise(dataset[:,4:7],0.00523)      # wx wy wz

    # dataset = remove_outliers(dataset,174.25,0.1) # Dataset, mass , % you want to cutoff Fz

    # dataset = z_score_outlier_detection(dataset,2.) # Z-score outlier removal

    # dataset = normalize(dataset)  # Normalize dataset (MIXALIS METHOD)

    # dataset = standard(dataset)   # standarize dataset


    # USE THIS FOR STELIOS NORMALIZE METHOD
    # for i in range(dataset.shape[1]):
    #     dataset[:,i] = normalize(dataset[:,i],np.max(dataset[:,i]))

    labels  = dataset[:,-1]         # delete labels
    dataset = np.delete(dataset,-1,axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=43)
    y_train = to_categorical(y_train,num_classes=2)
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
    X_test  = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

    contact = lcd()
    humanoid = False
    robot = "ATLAS"
    contact.setConfiguration(robot, humanoid)
    contact.fit(X_train, y_train,  15 ,16, True)




    # PREDICTIONS
    predict_x = contact.predict_dataset(X_test) # 20% of the training data
    classes_x = np.argmax(predict_x,axis=1)
    conf = confusion_matrix(y_test,classes_x)
    print(conf)
    print("Stable accuracy = ", conf[0,0]*100/(conf[0,0]+conf[0,1]))
    print("Slip  accuracy = ", conf[1,1]*100/(conf[1,0]+conf[1,1]))

    # TEST FILANAMES
    test_datasets_filenames = ['ATLAS_7k_04ground_coul_vel.csv','ATLAS_10k_05ground_coul_vel.csv','NAO_4k_01ground_coul_vel.csv','NAO_5k_03ground_coul_vel.csv','NAO_7k_05ground_coul_vel.csv']


    for filename in test_datasets_filenames:
      if filename[0:5] == 'ATLAS':
        mass = 174.25
      elif filename[0:3] == 'NAO':
        mass = 5.19535

      unseen = read_dataset(filename)

      # Remove FLY data points
      unseen = remove_fly_data(unseen)

      unseen = remove_features([0,1,3,4,5],unseen)

      unseen[:,0:1] = add_noise(unseen[:,0:1],0.6325)      # Fz
      unseen[:,1:4] = add_noise(unseen[:,1:4],0.0078)       # ax ay az
      unseen[:,4:7] = add_noise(unseen[:,4:7],0.00523)      # wx wy wz

      # unseen = z_score_outlier_detection(unseen,5)
      # unseen = remove_outliers(unseen,mass,0.1)

      # # unseen = standard(unseen)
      # unseen = z_score_outlier_detection(unseen,2.)

      # unseen = normalize(unseen)
      # unseen = standard(unseen)

      unseen_labels  = unseen[:,-1]
      unseen = np.delete(unseen,-1,axis = 1)
      unseen = unseen.reshape(unseen.shape[0],unseen.shape[1],1)

      predict_x1 = contact.predict_dataset(unseen)
      classes_x1 = np.argmax(predict_x1,axis=1)
      conf1 = confusion_matrix(unseen_labels,classes_x1)
      print(conf1)
      print("Stable accuracy = ", conf1[0,0]*100/(conf1[0,0]+conf1[0,1]))
      print("Slip  accuracy = ", conf1[1,1]*100/(conf1[1,0]+conf1[1,1]))
