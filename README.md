# collaboration-and-competition
### Environment Details

![Tennis](https://github.com/karthikrajkumar/collaboration-and-competition/blob/master/images/tennis.png)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. **Thus, the goal of each agent is to keep the ball in play.**

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, **The agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).** Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode

## Solution to this Environment
The environment is considered solved, when the average (over 100 episodes) of those scores is at least **+0.5**.

# Environment Details
For this use case, we dont have to install Unity environment as it has already been built. please find the details specific to operating systems.

*    [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
*    [Mac OS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
*    [Win 32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
*    [Win 64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## Installation details
you would be requiring the following
*    Python 3.6 or above [click here](https://www.python.org/downloads/)
    
*    Jupyter notebook/ lab

    python3 -m pip install --upgrade pip
    python3 -m pip install jupyter
    
    jupyter notebook
    
*    Pytorch [click for installation](https://pytorch.org/)
*    Numpy & matplotlib
     
    pip install numpy
    pip install matplotlib
    
## How to run this ?
Open the Tennis python notebook (Tennis.ipynb) and start running cell by cell or run all.
*    Note: The Unity environment needs to be downloaded and in Tennis.ipynb the path to load the environment needs to be changed accordingly.
