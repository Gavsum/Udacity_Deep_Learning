# Face Generation Project
## Overview
For this project we attempted to generate realistic looking images of celebrities based on the MNIST-Celeb-A dataset. 

![Actual celebrity faces from the dataset](https://i.imgur.com/kXslc0S.png)

## Model
For this task we utilized a Generative Adversarial (GAN) network. The GAN architecture utilizes two seperate networks: the generator, and the discriminator. The discriminator part of the GAN is tasked with attempting to classify data given to it, in this case checking whether or not an input image seems to be a valid celebrity face. The other half of the network, the generator, is sort of the inverse of the discriminator which attempts to predict features that are likely to appear in a piece of input given some label. 

The novel idea of a GAN is to feed the output of the generator network into the input of the discriminator network. The loss of the generator is how many of its output images the discriminator detects to be non-members of the celebrity set. The goal is to yield a generator network that outputs data which is very difficult for the discriminator to descern from ground trut input (real celebrity faces in this case).
![](https://i.imgur.com/ToWkVbn.png)
## Results
The resulting faces were less than amazing but the model reached the required loss values specified by the instructors.
![](https://i.imgur.com/Fn60zZP.png)
