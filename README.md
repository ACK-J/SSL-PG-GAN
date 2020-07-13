# SSL-PG-GAN
### A Semi-Supervised Progressively Growing Generative Adversarial Network

This project builds mainly off of the code presented in [Progressive Growing of GAN's for Improved Quality, Stability and Variation](https://arxiv.org/pdf/1710.10196.pdf)
 by Kerras et al. ICLR 2018. Inspiration also came from OpenAI's [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf) which details semi-supervised implementations of GANs.
 
My research incorporates semi-supervised learning alongside progressively growing training to achieve a model that can utilize both. Thus we do not need to worry about an overabundance of costly labeled data and are still abile to generate high definition images.  



# Installation 
Go to to get the latest version of anaconda for Python3 
it should download a shell script

https://www.anaconda.com/products/individual#linux

Once downloaded `chmod +x <The file you just downloaded>`

Then execute the file with either `./Anaconda3-2020.02-Linux-x86_64.sh` or `bash </path/to/file.sh>`

Complete the installation with all default settings 

Once complete open a new terminal so the changes take effect. 

**Make sure you have git installed**

`sudo apt update`

`sudo apt install git`

`sudo apt install build-essential`

`git clone https://github.com/ACK-J/SSL-PG-GAN.git`

Open up the file `environment.yml` in any text editor you want. Go all the way to the bottom of the file and you should see the line `prefix: /home/USERNAME/anaconda3/envs/ssl-pggan` Please change where it
says `USERNAME` to be your current username. Once that's done save and quit.

Next, run the following command to create the environment

`conda env create -f environment.yml`
- This could take a little while for everything to install depending on your internet connection

Once complete you will be able to start the environment with the following command

`conda activate ssl-pggan` 

Now you are in an environment with everything you need to run SSL-PGGAN

# Creating a dataset

You should have a dataset in mind that you want to use but if you don't I am providing the one I used to test
below so you can follow along and get something running 

https://drive.google.com/file/d/13OHIv7A_AFbKWe_URwbyOvEKRnPovrAq/view?usp=sharing

- Once downloaded place into the root of the SSL-PG-GAN folder
```
---------------------------------------------
The training data folder should look like : 
<SSL-PG-GAN Root Folder>
                |--Your DataFolder Named Anything
                        |--Unlabeled
                                |--image 1
                                |--image 2
                                |--image 3 ...
                        |--Labeled
                                |--Class 1
                                        |--image 1
                                        |--image 2
                                        |--image 3 ...
                                |--Class 2
                                        |--image 1
                                        |--image 2
                                        |--image 3 ...
                                |--Class 3 ...
---------------------------------------------
```

**Please note that every image must:**

    1. Have the same width and height
    2. Image resolution must be a power-of-two
    3. Image must be either grayscale or RGB
    
Example Image Size: 3 X 512 X 512

To create the cat/dogs dataset run the following command
`python3 dataset_tool.py <Labeled dir> <Unlabeled dir> 2> /dev/null`
- Make sure to use full paths for both Labeled and Unlabeled datasets.
- The reason that I send stderr to /dev/null is because the version of tensorflow has very noisy deprecation warnings which are annoying. 
- Now if you look in the `Labeled` and `Unlabeled` directories you should see a bunch of .tfrecord files

# Configuration
Edit `config.py` to see all of the options available. 
You must change two lines in the configuration file before training the model. 
They are both commented with `# CHANGE ME` and are the paths to the parent directory of the labeled and unlabeled data folders. This should normally just be the path to the 
SSL-PG-GAN folder.

Other configuration changes include changing the amount of Nvidia GPU's used.
- Note if you want to select multiple GPU's make sure the environment variable
CUDA_VISIBLE_DEVICES is set. You can do this by typing `echo $CUDA_VISIBLE_DEVICES`

You can view the GPU's available to you by typing `nvidia-smi` or `nvidia-smi -l`

You can change the GPU's selected by typing `export CUDA_VISIBLE_DEVICES=0,1,2,3`

_Make sure the amount of GPU's selected matches the uncommented line in the config.py file_

# Training
`python3 train.py`
- This will most likely take a while even with 4 GPU's running. I recommend running 
the training using `screen` or `tmux` so that if you lose connection via SSH you won't
have to restart the training.  

# Evaluating
`tensorboard --logdir results/000-ssl-pgan-sslpggan-preset-v2-4gpus-fp32/ --port 8888`
- Do not just copy and paste. Replace the `results/000-ssl-pgan-sslpggan-preset-v2-4gpus-fp32/` with the results directory you want to view.
- Once tensorboard boots up it will give you a URL to go to an address such as... `TensorBoard 1.14.0 at http://192.168.10.5:8888/ (Press CTRL+C to quit)`
 - Your IP-address will be different

# Testing
 `python3 test_discriminator.py <FULL path to out-of-sample image directory> <id of training round>`
- You can find the training round by going to `results/` directory. The number at the beginning of each directory identifies each training session. Make sure to use the entire 3 digit number ex) 000 or 001 etc...

## Comments and Tips
- It is not advised to run this code in a VM or a docker container due to the fact that it is tricky to pass graphics cards into them and have them function properly. This
is not to say that it isn't possible just that it may be time-consuming. 

