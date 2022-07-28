import numpy as np
import pandas as pd
import math
import random

from time import perf_counter
from numpy import mean,std
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from lcd import lcd
import matplotlib.pyplot as plt
# Parameters
humanoid = True
robot = "ATLAS"
noise = True    # Add more noise
prefix = 'data/'


def read_dataset(filename):
  df = pd.read_csv(filename,engine = 'python')
  dataset = df.values
  dataset = dataset.astype('float32')

  return dataset


# Adds gaussian noise to data
def add_noise(data,std):
  mu = 0  # mean and standard deviation
  s = np.random.normal(mu, std, data.shape[0])
  for f in range(0,data.shape[1]):
    for i in range(0,data.shape[0]):
      data[i,f] = data[i,f] + s[i]
  return data


# Normalize stelios
def normalize(din, dmax):
    if(dmax != 0):
        dout =  np.abs(din/dmax)
    else:
        dout =  np.zeros((np.size(din)))
    return dout

# Removes list of features ( for point feet robots)
def remove_features(features_to_remove,dataset):
  dataset = np.delete(dataset,features_to_remove,axis=1)
  return dataset


def remove_outliers(dataset,labels):
  # Outlier removal. (due to fall data spikes)
  feature_mean = []  # contains the mean value for every feature
  feature_std  = []  # contains the standard deviation of every feature

  for i in range(dataset.shape[1]):
    feature_mean.append(np.mean(dataset[:,i]))
    feature_std.append(np.std(dataset[:,i]))

  # identify outliers
  cut_off     = []
  lower_bound = []
  upper_bound = []
  num_std = 3       # how many sigmas

  for i in range(dataset.shape[1]):
    cut_off.append(feature_std[i]*num_std)
    lower_bound.append(feature_mean[i]-cut_off[i])
    upper_bound.append(feature_mean[i]+cut_off[i])

  outliers = []    # stores the indexes where the outliers are
  for i in range(dataset.shape[0]):
    for j in range(dataset.shape[1]):
        if (dataset[i,j] <= lower_bound[j]) or (dataset[i,j] >= upper_bound[j]):
          outliers.append(i)
          continue
  before_del = dataset.shape[0]
  # Delete the outliers from dataset
  dataset = np.delete(dataset,outliers,axis=0)

  labels  = np.delete(labels,outliers,axis=0)

  # print( "Removed -> ",  before_del- dataset.shape[0], " data samples out of ", before_del)
  return dataset,labels


# Merges no_contact and slip labels -> UNSTABLE contact
def merge_slip_with_fly(ls):
    for i in range(ls.shape[0]):
        if ls[i] == 2:
            ls[i] = 1
    return ls




if __name__ == "__main__":
    dataset = read_dataset(prefix + 'ATLAS_21k_02ground.csv')

    labels  = dataset[:,-1]         # delete labels
    dataset = np.delete(dataset,-1,axis = 1)

    dataset, labels = remove_outliers(dataset,labels)

    labels = merge_slip_with_fly(labels) # Merge slip labels with no_contact labels = Unstable contact

    dataset, labels = remove_outliers(dataset,labels)

    if humanoid:
        # add gaussian noise
        if noise == True:
            dataset[:,:3]  = add_noise(dataset[:,:3],0.6325)       # Fx Fy Fz
            dataset[:,3:6] = add_noise(dataset[:,3:6],0.03)        # Tx Ty Tz
            dataset[:,6:9] = add_noise(dataset[:,6:9],0.0078)      # ax ay az
            dataset[:,9:12] = add_noise(dataset[:,9:12],0.00523)   # wx wy wz
    else:
        # Remove features and add noise in case of point-feet robot
        dataset = remove_features([0,1,3,4,5],dataset)
        if noise == True:
            dataset[:,0:1] = add_noise(dataset[:,0:1],0.6325)       # Fz
            dataset[:,1:4] = add_noise(dataset[:,1:4],0.0078)       # ax ay az
            dataset[:,4:7] = add_noise(dataset[:,4:7],0.00523)      # wx wy wz


    #  Normalize
    for i in range(dataset.shape[1]):
        dataset[:,i] = normalize(dataset[:,i],np.max(abs(dataset[:,i])))



    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=43)
    y_train = to_categorical(y_train,num_classes=2)
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1])
    X_test  = X_test.reshape(X_test.shape[0],X_test.shape[1])

    contact = lcd()

    contact.setConfiguration(robot, humanoid)
    contact.fit(X_train, y_train,  15 ,16, True) # Train for 15 epochs


    # PREDICTIONS
    predict_x = contact.predict_dataset(X_test) # 20% of the training dataset
    classes_x = np.argmax(predict_x,axis=1)
    conf = confusion_matrix(y_test,classes_x)


    print(conf)
    print("Stable accuracy = ", conf[0,0]*100/(conf[0,0]+conf[0,1]))
    print("Slip  accuracy = ", conf[1,1]*100/(conf[1,0]+conf[1,1]))

    # TEST FILANAMES
    test_datasets_filenames = ['ATLAS_7k_04ground.csv','ATLAS_10k_05ground.csv','ATLAS_50k_mixedFriction.csv','NAO_13k_mixedFriction.csv','TALOS_50k_mixedFriction.csv']#'NAO_4k_01ground_coul_vel.csv','NAO_5k_03ground_coul_vel.csv','NAO_7k_05ground_coul_vel.csv']

    for filename in test_datasets_filenames:

      unseen = read_dataset(prefix + filename)

      unseenlabels = unseen[:,-1]
      unseen = np.delete(unseen,-1,axis = 1)

      print("SLIP",np.count_nonzero(unseenlabels == 2))
      print("STABLE",np.count_nonzero(unseenlabels == 0))
      print("FLY",np.count_nonzero(unseenlabels == 1))

      unseen, unseenlabels = remove_outliers(unseen,unseenlabels)

      unseenlabels = merge_slip_with_fly(unseenlabels)

      if humanoid:
        if noise:
          unseen[:,:3]   = add_noise(unseen[:,:3],0.6325)      # Fx Fy Fz
          unseen[:,3:6]  = add_noise(unseen[:,3:6],0.0316)     # Tx Ty Tz
          unseen[:,6:9]  = add_noise(unseen[:,6:9],0.0078)     # ax ay az
          unseen[:,9:12] = add_noise(unseen[:,9:12],0.00523)   # wx wy wz
      else:
        unseen = remove_features([0,1,3,4,5],unseen)
        if noise:
          unseen[:,0:1] = add_noise(unseen[:,0:1],0.6325)       # Fz
          unseen[:,1:4] = add_noise(unseen[:,1:4],0.0078)       # ax ay az
          unseen[:,4:7] = add_noise(unseen[:,4:7],0.00523)      # wx wy wz




      # Normalize
      for i in range(unseen.shape[1]):
          unseen[:,i] = normalize(unseen[:,i],np.max(abs(unseen[:,i])))

      unseen = unseen.reshape(unseen.shape[0],unseen.shape[1])

      start_counting = perf_counter()
      predict_x1 = contact.predict_dataset(unseen)
      stop_counting = perf_counter()

      print("Required prediction time per sample = ", (stop_counting-start_counting)/predict_x1.shape[0])

      classes_x1 = np.argmax(predict_x1,axis=1)
      conf1 = confusion_matrix(unseenlabels,classes_x1)

      # RESULTS
      print(filename)
      print(conf1)
      print("Stable accuracy = ", conf1[0,0]*100/(conf1[0,0]+conf1[0,1]))
      print("Slip  accuracy = ", conf1[1,1]*100/(conf1[1,0]+conf1[1,1]))

      # Save predictions
      # np.savetxt(path + filename,predict_x1,delimiter =',')
