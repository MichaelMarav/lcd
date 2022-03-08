# Legged Contact Detection (LCD)

LCD is a quality of contact estimation supervised deep learning framework for legged robots. It utilizes Force/Torque and IMU measurements to predict the probability of Stable Contact (SC) and Unstable Contact (UC). LCD also works with reduced features (Fz + IMU) in case the robot is point feet but the results are slightly worse on datasets with extremely low friction coefficient surfaces (< 0.07). Additionally, LCD makes cross-platform predictions, meaning it can predict the quality of contact on gaits of robot A even thought it was trained using data from robot B.

# Installation

Clone this repository in the workspace where your data (.csv) files are located. Modify the 'train.py' by defining the type of your robot (Humanoid or not) and change the filenames for train and test datasets. Then:

```bash
python3 train.py
```

## Useful tips

Train/Test: Dataset needs to consist of 12 features (in case Humanoid == True), namely Fx Fy Fz Tx Ty Tz ax ay az wx wy wz. If Humanoid == False then Fz ax ay az wx wy wz

If you want to extract the exact probability for stable contact remove the argmax from the predictions.

# Datasets

The datasets (.csv) that are given are self explanatory. Name of robot + number of samples + friction coeff. In the "mixed friction" datasets, the fiction coefficient varies from 0.03 to 1.2.

These data were collected from the raisim simulated environment.
