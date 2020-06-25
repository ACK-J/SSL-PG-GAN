SL-PG-GAN
A semi-supervised progressively growing generative adversarial network 


# Installation 
Go to to get the latest version of anaconda for Python3 
it should download a shell script

https://www.anaconda.com/products/individual#linux

Once downloaded `chmod +x <The file you just downloaded>`

Then execute the file with either `./Anaconda3-2020.02-Linux-x86_64.sh` or `bash </path/to/file.sh`

Complete the installation with all default settings 

Once complete open a new terminal so the changes take effect. 

**Make sure you have git installed**

`sudo apt update`

`sudo apt install git`

`sudo apt install build-essential`

`git clone https://github.com/ACK-J/SSL-PG-GAN.git`

Open up the file `environment.yml` in any text editor you want. Go all the way to the bottom of the file and you should 
see the line `prefix: /home/USERNAME/anaconda3/envs/ssl-pggan` Please change where it
says `USERNAME` to be your current username. Once thats done save and quit.

Next run the following command to create the environment

`conda env create -f environment.yml`
- This could take a little while for everything to install depending on your internet connection

Once complete you will be able to start the environment with the following command

`conda activate ssl-pggan` 

Now you are in an environment with everything you need to run SSL-PGGAN

# Creating a dataset

You should have a dataset in mind that you want to use but if you don't I am providing the one I used to test
below so you can follow along and get something running 

https://drive.google.com/file/d/13OHIv7A_AFbKWe_URwbyOvEKRnPovrAq/view?usp=sharing

- Once downloaded place into the root of the 

