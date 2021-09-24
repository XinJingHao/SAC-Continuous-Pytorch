

'''
Adapters of different Env, Used for better training.
See https://zhuanlan.zhihu.com/p/409553262 for better understanding.
'''


def Reward_adapter(r, EnvIdex):
    # For BipedalWalker
    if EnvIdex == 0 or EnvIdex == 1:
        if r <= -100: r = -1
    # For Pendulum-v0
    elif EnvIdex == 3:
        r = (r + 8) / 8
    return r

def Done_adapter(r,done,current_steps, EnvIdex):
    # For BipedalWalker
    if EnvIdex == 0 or EnvIdex == 1:
        if r <= -100: Done = True
        else: Done = False
    else:
        Done = done
    return Done

def Action_adapter(a,max_action):
    #from [-1,1] to [-max,max]
    return  a*max_action

def Action_adapter_reverse(act,max_action):
    #from [-max,max] to [-1,1]
    return  act/max_action





