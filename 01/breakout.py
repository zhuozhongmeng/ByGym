import tensorflow as tf
import numpy as np
import gym as gym
from collections import deque
import random
GAME = "Breakout-v4"
MEMORYSIZE = 6000
minisize = 300
GAMMA = 1

def ImgProcess(state):
    #第一个方式是抽取第一个层图像，等于使用灰度图
    state = state[32:192,0:160,:]
    #state = np.sum(state,axis=2)
    return state

class DQN ():

    def __init__(self,evn):
        self.get_memory_time = 0
        self.action_dim = evn.action_space.n
        self.session = tf.InteractiveSession()
        self.creat_net() #一开始就初始化，创建一个网络出来先

        self.memory = deque()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        self.m_reward = 0
        self.random_times = 0
        self.m_times = 0
        self.mm_reward = 1
        self.saver = tf.train.Saver()
    def get_weights(self, shape):
        weight = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(weight)

    def get_bias(self, shape):
        bias = tf.constant(0.01, shape=shape)
        return tf.Variable(bias)

    def creat_net(self): #创建tensorflow的图，用来直接逼出一个Q的网络价值函数
        self.img_input = tf.placeholder(dtype=tf.float32, shape=[None,160,160,3] )
        self.y_input = tf.placeholder(dtype=tf.float32,shape=[None])
        self.action_input = tf.placeholder(dtype=tf.float32,shape=[None, self.action_dim])

        w1= self.get_weights([8,8,3,32])
        b1 = self.get_bias([32])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(self.img_input,w1,[1,4,4,1],padding="SAME") + b1)
        conv1 = tf.nn.max_pool(h_conv1,[1,2,2,1],[1,2,2,1],padding="SAME")

        w2 = self.get_weights([4,4,32,64])
        b2 = self.get_bias([64])
        h_conv2 = tf.nn.relu(tf.nn.conv2d(conv1,w2,[1,2,2,1],padding="SAME") + b2)
        h_conv2 = tf.nn.max_pool(h_conv2,[1,2,2,1],[1,2,2,1],padding="SAME")


        w3 = self.get_weights([3,3,64,64])
        b3 = self.get_bias([64])
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2,w3,[1,1,1,1],padding="SAME") + b3)

        w_fc1 = self.get_weights([1600,512])
        b_fc1 = self.get_bias([512])
        conv3_flat = tf.reshape(h_conv3,[-1,1600])
        h_fc1 = tf.nn.relu(tf.matmul(conv3_flat,w_fc1) + b_fc1)

        w_fc2 = self.get_weights([512,self.action_dim])
        b_fc2 = self.get_bias([self.action_dim])

        self.Q_value = tf.matmul(h_fc1,w_fc2)+b_fc2  #直到这里，拿到的只是一个图像的识别结果抽象，带action的二维矩阵 当作是价值函数
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices=1) #这个是动作价值函数
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))  #   y_input 就是最佳策略得分，就是回报，来自于马尔可夫过程结果，这里就是让输出不断的毕竟最佳策略得分

        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)
        print("创建了一个网络")
    def reload (self):
        self.saver.restore(self.session, 'breakout/model.ckpt')
        print("读取记忆")
    def training(self):
        minibatch = random.sample(self.memory,minisize)  #这里是利用随机库，在记忆中，随机抽取一定数量minisize = 10 的记忆。然后等待下一步使用。
        mini_state =  [data[0] for data in minibatch]
        mini_action = [data[1] for data in minibatch]
        mini_next_state = [data[2] for data in minibatch]
        mini_reward = [data[3] for data in minibatch]
        mini_done = [data[4] for data in minibatch]
        total_Q = []
        next_Q_value = self.Q_value.eval(feed_dict={self.img_input:mini_next_state})

        for i in range(minisize):
            if mini_done[i]:
                total_Q.append(mini_reward[i])
                #print("游戏失败")
            else:
                total_Q.append(mini_reward[i] + GAMMA * np.max(next_Q_value[i]))
                #print("记录贝尔曼，",np.argmax(next_Q_value[i]))
                #print(np.shape(total_Q))
        self.optimizer.run( feed_dict={
            self.img_input:mini_state,
            self.action_input:mini_action,
            self.y_input:total_Q
        })
    def get_greedy_action(self,state):
        state = np.reshape(state,[1,160,160,3])
        action =  self.Q_value.eval(feed_dict={self.img_input:state})
        return action

    def __text__(self):
        print("ces")

    def get_action(self,state):
        if random.random() > 0.2:
            action = np.argmax(self.get_greedy_action(state))
            self.m_times += 1
        else:
            action = random.randint(0,self.action_dim - 1)
            self.random_times += 1
        return action

    def percieve(self,state,action,next_state,reward,done,now_times):

        action_index = np.zeros(self.action_dim)
        action_index[action] = 1
        self.get_memory_time += 1
        self.memory.append([state,action_index,next_state,reward,done,now_times])

        if len(self.memory) >minisize:
            self.memory.popleft()
        if self.get_memory_time % 4000 == 0:
            print("随机次数",self.random_times,"计算次数",self.m_times,"训练占比", self.m_times / (self.m_times + self.random_times))
            self.random_times = 0
            self.m_times = 0
            print("总得分",self.m_reward,"开始训练")
            for t in range(300):
                self.training()
                if t % 50 == 0 :
                    print("训练了",t,"次")
            self.saver.save(self.session, 'breakout/model.ckpt')
            print("训练完成并保存成功")
            self.mm_reward = self.m_reward
            self.m_reward = 0


        #print("这里存储记录")

def  main():
    evn = gym.make(GAME)
    agent = DQN(evn)
    agent.reload()
    init_state = evn.reset()
    init_state  = ImgProcess(init_state)
    for times in range(100000000):
        evn.render()
        nowtime_reward = 0
        if times == 0:
            state = init_state  #初始化的时候的state

        #print(evn.state_space.n)
        action = agent.get_action(state)

        next_state,reward,done,_ =  evn.step(action)
        next_state = ImgProcess(next_state)
        agent.percieve(state,action,next_state,reward,done,times)
        state = next_state
        agent.m_reward += reward
        if reward > nowtime_reward:
            nowtime_reward = reward

        if done :
            evn.reset()

if __name__ == '__main__':

        main()
