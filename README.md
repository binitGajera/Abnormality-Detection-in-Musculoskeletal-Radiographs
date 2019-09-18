# Abnormality Detection in Musculoskeletal Radiographs

This project is basically a part of the Musculoskeletal Radiographs dataset competition provided by [Stanford](https://stanfordmlgroup.github.io/competitions/mura/). It uses a pre-trained DenseNet169 model provided by [Pytorch](https://pytorch.org/hub/pytorch_vision_densenet/), and trains again on the dataset so that the featureset of the original dataset is not lost but realigned to the current dataset. After which, the output of the model would be whether the patient's radiographs(there would be multiple radiographs for a single patient) are normal or abnormal.

So the main task at hand for this project is to predict if a patient's hand radiograph is **normal** or **abnormal**.

## Short Description

The main file that runs the code and would have to be executed to run the project is *main.py*, given if the directory structure is not changed the code should run successfully. The only constraint to the complete project and the one here on Github is that this project only has 10% of the dataset that I originally used to obtain my results of training and testing.

The complete dataset although can be downloaded from: [MURA](https://stanfordmlgroup.github.io/competitions/mura/)

### Libraries Used

The code is written in Python, hence several of the libraries will be required in order to execute the code. Some of them are as follows:

```
- torch
- torchvision
- tqdm
- pandas
- Binit
- sys
```

## Built With

* [Google Colab](https://colab.research.google.com) - The platform to run the code
* [Spyder](https://www.spyder-ide.org/) - IDE - To code the project
* [Pytorch](https://pytorch.org/) - Machine Learning library to create the model

## Report

Please feel free to refer to the [Report](https://umbc.app.box.com/s/idb3k8wvoahu3s4kib6mly4f8juewyin) to learn more about the model architecure, how machine learning was used for this project, and view the results obtained from the project.
