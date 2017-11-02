# Malmo Deep Q-Learning #

Malmo is a platform provided by Microsoft to design, build, and evaluate an intelligent agent. Using deep Q-learning, it has been tried to design an agent that can find its way in a maze.

## Getting Started

These instructions will get you a copy of the project up and run on your local machine for development and testing purposes. See instruction to deploy the project on a live system.
 
### Prerequisites

You need to know some basic concepts in reinforcement learning and neural networks. Try to familiarize yourself with neural nets, backpropagation, reinforcement learning, Q-learning, and memory replay. For this project, python 2.7 and macOS Sierra are used. Also, to implement neural nets, the Pytorch framework is considered.

On MacOSX, currently only the system python is supported, so we use the python in */usr/bin/python* to install packages and run the code.

#### Installing Malmo

To run this code you need to first install Malmo. Follow the instruction below to run the agent (this instruction is only for Mac users. For other operating systems you can refer to this [repository](https://github.com/Microsoft/malmo) for more information. ) :

1. First download [Malmo-0.17.0-Mac-64bit](https://github.com/Microsoft/malmo/releases).
2. Unzip the file and put it in your working directory.
3. Open a terminal and change your directory to Minecraft folder:
```
cd Path_to_Your_Working_Directory/Malmo-0.17.0-Mac-64bit/Minecraft
```
4. Run the script called launchClient.sh and wait until the installation finishes. As long as, you can see the main menu the installation has finished and you don't need to wait for 95% progress indicator in the terminal to become 100%:
```
./launchClient.sh
```

#### Installing required pachages

1. Installing Pytorch:
```
/usr/bin/python -m pip install http://download.pytorch.org/whl/torch-0.2.0.post3-cp27-none-macosx_10_7_x86_64.whl 
```
2. Installing Numpy:
```
/usr/bin/python -m pip install numpy 
```
3. Installing matplotlib:
```
/usr/bin/python -m pip install matplotlib 
```

#### Initialize some variables in myagent.py

1. Clone this repository into :

```
cd Path_to_Your_Working_Directory/Malmo-0.17.0-Mac-64bit/Python_Examples

git clone https://github.com/otoofim/Malmo.git
```
2. Open myagents.py and change line 727. It is a variable called DEFAULT_MALMO_PATH which holds Malmo directory. So, just change it to where you have already put Malmo:
```
DEFAULT_MALMO_PATH = 'Path_to_Your_Working_Directory/Malmo-0.17.0-Mac-64bit'
```
3. You also need to change line 728 where there is a variable called DEFAULT_AIMA_PATH. Put the address where you have put Aima:
```
DEFAULT_AIMA_PATH = 'Path_to_Your_Working_Directory/aima-python/'
```

### Run the agent

for running the myagents.py there are some parameters that can be passed to the main function. The list of the parameters are here:

1. "-a" , "--agentname": select agent type ("Helper", "Realistic", "Random")
2. "-t" , "--missiontype": mission type ("small","medium","large")
3. "-s" , "--missionseedmax": it is used to randomly generate a maze (integer)
4. "-n" , "--nrepeats": number of episods (if stochastic behavior)

All the options above are optional and in the case, they are not provided default values will be used.

While Minecraft is being run, open a new terminal and enter the following commands in the terminal:
```
cd Path_to_Your_Working_Directory/Malmo-0.17.0-Mac-64bit/Python_Examples
/usr/bin/python myagents.py -a Realistic
```
