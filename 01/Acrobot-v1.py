import tensorflow as tf
import gym as gym
import numpy as np
import random as random
from collections import deque


#GAME = "CartPole-v0"
#GAME = "Acrobot-v1"
#GAME = "Amidar-ramDeterministic-v4"
GAME = "RoadRunner-v0"
#GAME = "PhoenixDeterministic-v4"



class DQN():
    Q_value = None
    state_input = None
    session = tf.Session()
    def __init__(self,env):
        #self.saver = tf.train.Saver()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.memory = deque()
        self.creat_network()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        self.randomtimes = 0
        self.traintimes = 0
    def creat_network(self):
        self.action_input = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32)
        self.y_input = tf.placeholder('float', [None])
        w1 = tf.Variable(tf.random_normal([self.state_dim,20]))
        b1 = tf.Variable(tf.random_normal([20]))
        w12 = tf.Variable(tf.random_normal([20, 20]))
        b12 = tf.Variable(tf.random_normal([20]))
        #w22 = tf.Variable(tf.random_normal([20, 20]))
        #b22 = tf.Variable(tf.random_normal([20]))
        w2 = tf.Variable(tf.random_normal([20,self.action_dim]))
        b2 = tf.Variable(tf.random_normal([self.action_dim]))
        self.state_input =  tf.placeholder('float',[None,self.state_dim])
        h_layer1  = tf.nn.relu(tf.matmul(self.state_input,w1)+b1)
        h_layer12 = tf.nn.relu(tf.matmul(h_layer1, w12) + b12)
        #h_layer22 = tf.nn.relu(tf.matmul(h_layer12, w22) + b22)
        self.Q_value = tf.matmul(h_layer12,w2)+b2
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.00005).minimize(self.loss)

    def get_action(self,state):
        self.randomtimes += 1
        if self.randomtimes > 90000:
            self.randomtimes = 90000
        if random.random()>1-0.00001*self.randomtimes:
            return self.greedy(state)
        else :
            return random.randint(0,self.action_dim-1)



    def greedy(self,state):
        Q_val_output =  self.session.run(self.Q_value,feed_dict={self.state_input:[state]})[0]
        #print(np.argmax(Q_val_output))
        return np.argmax(Q_val_output)

    def persaver(self,state,action,reward,next_state,done):
        action_index = np.zeros(self.action_dim)
        action_index[action]= 1
        self.memory.append((state,action_index,reward,next_state,done))
        if len(self.memory)>10000:
            self.memory.popleft()

        if len(self.memory)%50 ==0:
            self.train()
            self.traintimes +=1

    def train(self):
        minibatch = random.sample(self.memory,50)
        mini_state = [data[0] for data in minibatch]
        mini_action_index =[data[1] for data in minibatch]
        mini_reward = [data[2] for data in minibatch]
        mini_nextstate =[data[3] for data in minibatch]
        mini_done = [data[4] for data in minibatch]
        mini_y = []
        Q_next_state = self.session.run(self.Q_value,feed_dict={self.state_input:mini_nextstate})

        for i in range(len(minibatch)):
            if mini_done[i]:
                mini_y.append(mini_reward[i])
            else:
                mini_y.append(mini_reward[i]+ np.max(Q_next_state[i]))


        self.session.run([self.optimizer,self.loss],feed_dict={
            self.state_input:mini_state,
            self.action_input:mini_action_index,
            self.y_input:mini_y
        })

def main():
    env = gym.make(GAME)
    agent = DQN(env)
    state = env.reset()
    print(state)
    m_reward = 0

    for rounds in range(100000):

        for times in range(150000):
            env.render()
            action = agent.get_action(state)
            #print(action)
            next_state, reward, done, _ = env.step(action)
            #print(next_state, reward, done)
            #if reward == 0:
                #print(reward)
            m_reward += reward

            agent.persaver(state,action,reward,next_state,done,)
            state = next_state
            if done:
                env.reset()
                print(m_reward,len(agent.memory),agent.traintimes,agent.randomtimes)
                m_reward = 0
                agent.traintimes = 0
                break;




if __name__ == '__main__':
    main()
