# ERA Session 5 

Session 5 Python Notebook leverages the basic MNIST dataset to train a neuran network model. The code which is used to train the model is written using Pytorch.
PyTorch is an open source machine learning (ML) framework based on the Python programming language and the Torch library. Torch is an open source ML library used for creating deep neural networks and is written in the Lua scripting language. It's one of the preferred platforms for deep learning research. The framework is built to speed up the process between research prototyping and deployment.

The PyTorch framework supports over 200 different mathematical operations. PyTorch's popularity continues to rise, as it simplifies the creation of artificial neural network models. PyTorch is mainly used by data scientists for research and artificial intelligence (AI) applications. PyTorch is released under a modified BSD license.

PyTorch was initially an internship project for Adam Paszke, who at the time was a student of Soumith Chintala, one of the developers of Torch. Paszke and several others worked with developers from different universities and companies to test PyTorch. Chintala currently works as a researcher at Meta -- formerly Facebook -- which uses PyTorch as its underlying platform for driving all AI workloads.


# The Core components of the program includes 
1. utils file 
2. model file 
3. S5 Python notebook
4. Readme file 

This program is completely modularize and it can be extended for all future programs 

# How to use the Program 
Clone the reposirty "https://github.com/nkanungo/ERA_1.git" or download the repository 

**utils file **

Utils file contains all the utility functions required for the main program . It contains 
  **get_train_transform** - Transofrm the training data 
  1. Center Crop
  2. Random Rotation
  3. Normalization
  **get_test_transform** - Transofrm the test data 
   1. Normalization
**get_train_loader**
1. Download the training data 
2. Create Train loader

**get_test_loader**
1. Download the test data 
2. Create Test loader

**Train Method **

This function takes a model, device, train_loader, optimizer, and criterion as inputs. It trains the model using the provided data and parameters, and returns lists of training losses and accuracies for each epoch.

The function first initializes empty lists to store the training losses and accuracies. It then sets the model to training mode and creates a progress bar for the training data loader.

Inside the training loop, it performs the following steps for each batch of data:

Moves the data and target tensors to the specified device.
Resets the gradients of the optimizer.
Performs a forward pass through the model to predict the output for the input data.
Calculates the loss between the predicted output and the target using the specified loss function.
Accumulates the training loss for the current batch.
Computes gradients of the loss with respect to the model parameters.
Updates the model parameters using the optimizer.
Updates the number of correct predictions and the number of processed samples.
Updates the progress bar description with the current loss and

**Test Method**

This function is used to evaluate a trained model on the test data. It takes the model, device, test_loader, and criterion as inputs, and returns lists of test losses and accuracies for each evaluation.

The function first initializes empty lists to store the test losses and accuracies. It then sets the model to evaluation mode.

Inside the evaluation loop, it performs the following steps for each batch of data:

Moves the data and target tensors to the specified device.
Performs a forward pass through the model to predict the output for the input data.
Accumulates the test loss for the current batch using the specified loss function.
Updates the number of correct predictions.
After iterating over all batches, the function calculates the average test loss and test accuracy. It appends these values to the respective lists.

Finally, the function prints the evaluation results, including the average test loss and the accuracy percentage.

The Program trains and provide approximately 99.3% accuracy .

**Parameters used by the code
**



![image](https://github.com/nkanungo/ERA_1/assets/40553830/ece9af0a-1127-45e7-b832-478af20c5105)



**The training and Testing visualization **



![image](https://github.com/nkanungo/ERA_1/assets/40553830/ec8ccb8b-06e0-4c01-9b05-cac18387551f)

![image](https://github.com/nkanungo/ERA_1/assets/40553830/4804531a-340b-4817-a13b-9fd285d44e63)


**Contribution **

Nihar Kanungo ( Student)

The School of AI ( Mentor)

