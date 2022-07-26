# Working Dirs
This folder is the output of the training.
Here, I attached the tf_logs for tensorboard graphs visualization, and the log files.
Usually in this folder you will have the .pth file as well. However, since it was too heavy, I preffered to keep it inside the Google Drive and mount the directory instead.
Link for .pth: https://drive.google.com/file/d/1eb0V3ZxJBX1xRU6oIOGKx9uFDydhxtG5/view?usp=sharing

# Config Files
Here you have the config yolox_l_8x8_300e_coco_airbus.py, however

The configs can inherit from other primitive (that inhirit from base) configs, while max of inheritance level is 3. 
It is recommended to inherit from existing methods, when they share structure with the desired model.
For example, inherit the basic Faster R-CNN structure, then modify the necessary fields there.
Config naming style: {model}[model setting]{backbone}{neck}[norm setting][misc][gpu x batch_per_gpu]{schedule}{dataset}\
while {xxx} is required field and [yyy] is optional.

There are 4 basic component types under config/base:
1. model
2. dataset
3. schedule
4. default_runtime

------------------------------------------------------------------------------------------------------------------------

## 1. Model

Many models for many tasks can composed by the following 3 parts.
For each part- there are many of-the-shelve open-source architectures. Thus, one can decide how to build these parts as a puzzle.
Moreover, one can decide to use pre-trained models- that is the ability to start from a much better starting point- 
a machine that learned already the basics very well- based on many images and high computational efforts- no need to start from scratch, right?!

**Backbone**: The most basic part which extract general features from images. Well known CNN backbones is for example ResNet50/101. This part is usually taken with the pre-trained weights.

**Neck**: Used for the creation of pyramid feature. It helps the module on scaling factor of detected objects which are of the same nature but different scales. An example for

**Head**: Designed specifically per task- classification/ segmentation/ etc. usually this part will have the most changed while training on a new data-source/ task.

## 2. Dataset

contains the config of the data for train, test and validation.
including the pipelines that are augmenting the data.
I added a mini test-set with a few examples in the directory "mini_test"..
just so you can see how the data and the annotations looks like.

## 3. Scheduler and Runtime Parameters - 'Hyper-Parameters'
These are more parameters- containing the optimizer parameters, logs, num of epochs, learning rate, warmup phase, etc...

---
**Note #1**: 
There are also networks that doesn't work this way, with the 3 parts, but sometimes still can be composed as puzzle...
---
**Note #2**: The academy/ research centers are usually those in-charge of developing a new-model or concepts. 
Usually the diffrences between the performances of one method in compare to the others are minors, or specifit to the task and it's prior knowledge. 
However, using the MMCV framework, one can easily be adjusted to these changes, as the format are in well known structure and shell be generic enough for the adaptation.
---