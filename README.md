# Legged Contact Detection (LCD)

LCD is a quality of contact estimation supervised deep learning framework for legged robots. It utilizes Force/Torque and IMU measurements to predict the probability of Stable Contact (**SC**) and Unstable Contact (**UC**). LCD also works with reduced features (Fz + IMU) in case the robot is point feet but the results are slightly worse on datasets with extremely low friction coefficient surfaces (< 0.07). Additionally, LCD makes cross-platform predictions, meaning it can predict the quality of contact on gaits of *robot A* even thought it was trained using data from *robot B*. It can also run real-time with

## Installation of LCD

Clone this repository in your workspace. The default robot type for training and testing is *Humanoid*, meaning you'll need to provide a dataset with 12 features and 1 label per sample (0 or 1). The dataset will need to have the following structure:

| Fx | Fy | Fz | Tx | Ty | Tz | ax | ay | az | wx | wy | wz | label |, for every sample.

In case you want to use the *reduced lcd* for point feet robots, just change *Humanoid = False* and provide the reduced dataset:

| Fz | ax | ay | az | wx | wy | wz | label |

You can also toggle on/off the gaussian noise by changing **noise = True/False** and if you want to extract  probabilities P(UC) or P(SC) instead of labels 0 or 1, you can comment the *argmax* from the prediction part.


 Finally modify the 'train.py' and enter the absolute path to the files where the train and test datasets are located. Then:

```bash
python3 train.py
```

The prediction time per sample is 1e-5, thus it can make prediction real-time, but this is not yet implemented and it is a work in progress.

## Setup raisim simulator
In case you want to setup the raisim simulator with either ATLAS or NAO. 

![Screenshot](src/images/atlas.png)

You will need to install the following packages:

1. https://github.com/mrsp/raisim_ros
2. https://github.com/mrsp/lipm_control
3. https://github.com/mrsp/lipm_motion
4. https://github.com/mrsp/whole_body_ik

## Provided Datasets

The datasets (.csv) that are given are self explanatory. Name of robot + number of samples + friction coeff. In the "mixed friction" datasets, the fiction coefficient varies from 0.03 to 1.2. The labels are either 0 (stable contact), 1 (no_contact) and 2 (slip). Labels 1 and 2 are merged into one class by lcd namely UC. These samples were collected from the raisim simulated environment and the labels were extracted by using the force and also by utilizing the ground-truth linear and angular velocity of the foot. Each dataset consists of omnidirectional walking gaits of the named robot with a sampling rate of 100 hz



## Results

Walking in extremely low coefficient surface scenarios causes the robot to slip but since it transfers its weight to the slipping foot, the vertical ground reaction force is high-valued and thus mainstream algorithms for predicting the contact state fail.

The prediction of LCD on an experiment with an ATLAS robot walking on a surface with friction coefficient below 0.1 are represented in the following figure. The top labels mean stable contact and the bottom mean unstable contact (Slip + fly):

![Screenshot](src/images/lcdexplained.png)


The first two steps are on normal friction coefficient and thus the labels behave as expected. Meanwhile the third and forth steps are when the robot slips even though the GRF values are the same as before.

