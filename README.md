# To run create the folders and the files run this script file which is the template.sh and then in the git bash run this command

sh template.sh


# To create the virtual env for the project i first check wheather i have the conda installed in the system by running the command 

conda --version 


# if not already installed then install it from the chrome from \
https://www.anaconda.com/download/success

# And then add the paths to the env there will be three path ask chat gpt for that 

# And then run this command to creat the virtual env
conda create -n medibot python=3.10 -y

# And now run this to activate the v env
conda activate medibot
# sometime there is the possibility of this command not running in the git bash so run this command in the anaconda prompt that u need to install if it is not already installed in ur system 


# run this to install pip install all the requirements of the projects 
pip install -r requirements.txt
