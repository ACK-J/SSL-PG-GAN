# SSL-PG-GAN
### A Semi-Supervised Progressively Growing Generative Adversarial Network
⚠THIS REPOSITORY IS UNDER ACTIVE DEVELOPMENT⚠

This project builds mainly off of the code presented in [Progressive Growing of GAN's for Improved Quality, Stability and Variation](https://arxiv.org/pdf/1710.10196.pdf)
 by Kerras et al. ICLR 2018. Inspiration also came from OpenAI's [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf) which details semi-supervised implementations of GANs.
 
My research incorporates semi-supervised learning alongside progressively growing training to achieve a model that can utilize both. Thus we do not need to worry about an overabundance of costly labeled data and are still abile to generate high definition images.  



# Installation 
Go here to get the latest version of anaconda for Python3 
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

Open up the file `environment.yml` in any text editor you want. Go all the way to the bottom of the file and you should see the line `prefix: /home/USERNAME/anaconda3/envs/ssl-pggan` Please change where it says `USERNAME` to be your current username. Note that I am assuming that you installed anaconda in your home folder (If this is not the case replace the path with where anaconda was installed). 

Once that's done save and quit.

Next, run the following command which will construct an environment pre-made with all the dependencies needed to run the SSL-PG-GAN.

`conda env create -f environment.yml`
- This could take a little while for everything to install depending on your internet connection


`conda activate ssl-pggan` 

Now you are in an environment with everything you need to run SSL-PGGAN

# Creating a dataset

You should have a dataset in mind that you want to use but if you don't I am providing the one I used to test
below so you can follow along and get something running 
- The images provided in the dataset below are already resized to 256x256 and are in the correct file structure

https://drive.google.com/file/d/13OHIv7A_AFbKWe_URwbyOvEKRnPovrAq/view?usp=sharing

- Once downloaded place, the unziped folder, into the root of the SSL-PG-GAN folder
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
    
Example Image Size: 3 X 256 X 256

### How to resize the images you already have?
Most likely your dataset of images will not be 256x256 or 512x512 or 1024x1024 etc... but I made an easy one-liner where you can resize an entire directory full of images!

First you are going to need to install `imagemagick`

`sudo apt-get install imagemagick -y`

The One-Liner:

`find <path to img folder> -iname '*.jpg' -exec convert \{} -verbose -resize 256x256! \{} \;`
- Find is a command that will list all of the contents of the directory you give it
- `-iname` says look for files and be case-insensitive.
- `*.jpg` will look for all files in the direcotry with a .jpg extension
   - If you are not using jpg files change the extension
- The `-exec` says pass all of the files you found to the following command
- The command uses `imagemagick` to resize all the images to be exactly `256x256` pixels and if they are smaller it stretches them until they are
   - If you do not want your pictures to be `256x256` then change it and make sure to leave the `!` where it is
   - The `\{}` is where the file names that the find command gets are placed.

**NOTE:This command will overwrite the files when it converts them to be the fixed size**

To create the `cat/dogs` dataset (Or any dataset of images) run the following command...

`python3 dataset_tool.py <Labeled dir> <Unlabeled dir> 2> /dev/null`
- Make sure to use full paths for both Labeled and Unlabeled datasets holding your images.
- The reason that I send stderr to `/dev/null` is because the version of tensorflow has very noisy deprecation warnings. 
- Now if you look in the `Labeled` and `Unlabeled` directories you should see a bunch of `.tfrecord` files

# Configuration
Edit `config.py` to see all of the options available. 

You must change two lines in the configuration file before training the model. 

They are both commented with `# CHANGE ME` and are the paths to the parent directory of the labeled and unlabeled data folders holding the `.tfrecord` files. This should normally just be the path to the 
SSL-PG-GAN folder.

Other configuration changes include changing the amount of Nvidia GPU's used.
- Note if you want to select multiple GPU's make sure the environment variable
`CUDA_VISIBLE_DEVICES` is set. You can do this by typing `echo $CUDA_VISIBLE_DEVICES`

You can view the GPU's available to you by typing `nvidia-smi` or `nvidia-smi -l`

You can change the GPU's selected by typing `export CUDA_VISIBLE_DEVICES=0,1,2,3` into your terminal.

_Make sure the amount of GPU's selected matches the uncommented line in the config.py file_

# Training
`python3 train.py`
- This will most likely take a while even with 4 GPU's running. I recommend running 
the training inside a `screen` or `tmux` session so that if you lose connection via SSH you won't
have to restart the training.  

# Evaluating
`tensorboard --logdir results/000-ssl-pgan-sslpggan-preset-v2-4gpus-fp32/ --port 8888`
- Do not just copy and paste. Replace the `results/000-ssl-pgan-sslpggan-preset-v2-4gpus-fp32/` with the results directory you want to view.
- Once tensorboard boots up it will give you a URL to go to such as... `TensorBoard 1.14.0 at http://192.168.10.5:8888/ `
 - Your IP-address will be different
 ### Alternatively, you could host it on `tensorboard.dev` so you can access it from anywhere instead of just locally.
 `pip install -U tensorboard`
 
 `tensorboard dev upload --logdir results/000-ssl-pgan-sslpggan-preset-v2-4gpus-fp32/ --name "SSL-PG-GAN" --description "First run!"`

# Testing
 ` python3 test_discriminator.py <FULL path to out of sample images> <id of training round> <pixels> <OPTIONAL: index of correct class>`
 
 `python3 test_discriminator.py /home/user/OutOfSample/Images/ 2 256 1`
- You can find the training round by going to `results/` directory. The number at the beginning of each directory identifies each training session. Make sure to use the entire 3 digit number ex) 000 or 001 etc...
- Make sure `export CUDA_VISIBLE_DEVICES=` is set to an open GPU

## Comments and Tips
- It is not advised to run this code in a VM or a docker container due to the fact that it is tricky to pass graphics cards into them and have them function properly. This
is not to say that it isn't possible just that it may be time-consuming. 

