# SAC-Continuous-Pytorch
I found the current implementation of Soft Actor Critic on continuous action space is somewhat complicated, which is hard to get start.  
And this is a **clean and robust Pytorch implementation of SAC on continuous action space**. Here is the result:  
  
![avatar](https://github.com/XinJingHao/SAC-Continuous-Pytorch/blob/main/imgs/result.jpg)  
All the experiments are trained with same hyperparameters recommended by [Haarnoja et al](https://arxiv.org/pdf/1812.05905.pdf).  
  
The gif below is a short record of the performance on BipedalWalkerHardcore-v3:
![avatar](https://github.com/XinJingHao/SAC-Continuous-Pytorch/blob/main/imgs/BipedalWalkerHardcore-v3.gif)  

## Dependencies
gym==0.18.3  
box2d==2.3.10  
numpy==1.21.2  
pytorch==1.8.1  
tensorboard==2.5.0 


## How to use my code
### Train from scratch
run **'python main.py'**, where the default enviroment is Pendulum-v0.  
### Change Enviroment
If you want to train on different enviroments, just run **'python main.py --EnvIdex 0'**.  
The --EnvIdex can be set to be 0~5, where   
'--EnvIdex 0' for 'BipedalWalker-v3'  
'--EnvIdex 1' for 'BipedalWalkerHardcore-v3'  
'--EnvIdex 2' for 'LunarLanderContinuous-v2'  
'--EnvIdex 3' for 'Pendulum-v0'  
'--EnvIdex 4' for 'Humanoid-v2'  
'--EnvIdex 5' for 'HalfCheetah-v2' 

P.S. if you want train on 'Humanoid-v2' or 'HalfCheetah-v2', you need to install **MuJoCo** first.
### Play with trained model
run **'python main.py --EnvIdex 1 --write False --render True --Loadmodel True --ModelIdex 2800000'**, which will render the 'BipedalWalkerHardcore-v3'.  
### Visualize the training curve
You can use the tensorboard to visualize the training curve. History training curve is saved at '\runs'
### Hyperparameter Setting
For more details of Hyperparameter Setting, please check 'main.py'
### Reference
[Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf)

### Other RL algorithms by Pytorch can be found [here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch).
