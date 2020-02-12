from collections import namedtuple,deque
import random
import torch
from SumTree import SumTree
Transition = namedtuple('Transion', 
                        ('state', 'action', 'reward', 'next_state','done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory =deque(maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        sample=random.sample(self.memory, batch_size)
        batch=Transition(*zip(*sample))
        state,action,reward,next_state,done=map(torch.cat,[*batch])
        return state,action,reward,next_state,done
class N_step_ReplayMemory(object):
    def __init__(self, capacity,n_step,gamma):
        self.capacity = capacity
        self.memory =deque(maxlen=capacity)
        self.n_step_memory=deque(maxlen=n_step)
        self.n_step=n_step
        self.gamma=gamma
    def push(self, *args):
        self.n_step_memory.append(Transition(*args))
        if len(self.n_step_memory)>=self.n_step:
            state,action=self.n_step_memory[0][:2]
            reward=0
            i=0
            for transition in self.n_step_memory:
                _,_,r_t,next_state,done=transition
                reward+=r_t*self.gamma**i
                if done:
                    break
                i+=1
            self.memory.append(Transition(state,action,reward,next_state,done))
        return

    def sample(self, batch_size):
        sample=random.sample(self.memory, batch_size)
        batch=Transition(*zip(*sample))
        state,action,reward,next_state,done=map(torch.cat,[*batch])
        return state,action,reward,next_state,done

class PER(object):
    def __init__(self,capacity,n_step,gamma,alpha):
        self.capacity = capacity
        self.memory =deque(maxlen=capacity)
        self.n_step_memory=deque(maxlen=n_step)
        self.n_step=n_step
        self.gamma=gamma
        self.alpha=alpha
        self.tree=SumTree(capacity)
        self.empty=True
    def push(self,*args):
        self.n_step_memory.append(Transition(*args))
        if len(self.n_step_memory)>=self.n_step:
            if self.empty:
                priority=1
                self.empty=False
            else:
                priority=self.tree.tree[-self.capacity:].max()
            state,action=self.n_step_memory[0][:2]
            reward=0
            i=0
            for transition in self.n_step_memory:
                _,_,r_t,next_state,done=transition
                reward+=r_t*self.gamma**i
                if done:
                    break
                i+=1
            transition=Transition(state,action,reward,next_state,done)
            self.tree.add(priority, transition)
        return
    def sample(self,batch_size,beta):
        k_range=self.tree.total()/batch_size
        sum_p=self.tree.tree[0]
        states=[]
        actions=[]
        rewards=[]
        next_states=[]
        dones=[]
        indexes=[]
        IS_weights=[]
        p_min = self.tree.min_priority()/ sum_p
        w_max = (p_min *batch_size) ** (-beta)
        for i in range(batch_size):
            low= k_range * i
            high = k_range * (i + 1)
            s=random.uniform(low,high)
            index,priority,transition=self.tree.get(s)
            state,action,reward,next_state,done=transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            indexes.append(index)
            p_i=priority/sum_p
            w_i=(p_i*batch_size)**(-beta)
            IS_weight=w_i/w_max
            IS_weights.append(IS_weight)
        return torch.cat(states),torch.cat(actions),torch.cat(rewards),torch.cat(next_states),torch.cat(dones),indexes,IS_weights
    def update_priority(self,indexes,priorities):
        for index, priority in zip(indexes, priorities):
            self.tree.update(index,priority**self.alpha)
        return
            
            
        
            
        