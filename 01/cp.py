import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
import time

EXPLORE = 100000  # 训练次数

ENV_NAME = "CartPloe-v0"
EPISODE = 10000  # 演练次数
STEP = 300  # 每次的训练次数

class DQN():
    # 定义一个DQN的类

    # 首先我们建立一个DQN的类，再首先我们为DQN设置一些超参数
    GAMMA = 0.9999
    INITIAL_EPSILON = 0.5
    FINAL_EPSILON = 0.01  # 小正数
    MEMORY_SIZE = 10000
    BATCH_SIZE = 32

    # 以上是给到DQN使用的参数，具体使用等下需要一一的对应核实
    # 先定义一个空的动作记忆
    action_list = None

    state_input = None
    Q_value = None
    y_input = None
    optimizer = None
    cost = 0
    session = tf.Session()
    cost_history = []
    traintimes = 0

    # 定义一个初始化方法
    def __init__(self, env):
        self.randomtimes = 0
        self.randomarea = 0.5
        self.memory = deque()  # 定义一个经验的缓存数据，使用的是双向队列
        self.time_step = 0
        self.epsilon = self.INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]  # 观测状态空间的组数，数据的0号维度
        self.action_dim = env.action_space.n  # 动作空间组数

        self.action_list = np.identity(self.action_dim)  # 得到了一个2*2的单位矩阵
        self.creat_Q_network()
        self.create_training_method()

        # init session  #tensorflow的初始化；
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
    # 如我理解，这里只是一个简单的relu为激活函数的两层神经网络，用了TF作为框架，直接做了训练，输入数据是当前状态。输出数据为Q值。
    # 暂时没有对这个网络对输入输出的实际意义进行深入的理解
    def creat_Q_network(self):

        W1 = tf.Variable(tf.random_normal([self.state_dim, 20]))
        b1 = tf.Variable(tf.random_normal([20]))
        W2 = tf.Variable(tf.random_normal([20, self.action_dim]))
        b2 = tf.Variable(tf.random_normal([self.action_dim]))
        print(W1, W2)
        self.state_input = tf.placeholder('float', [None, self.state_dim])
        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        self.Q_value = tf.matmul(h_layer, W2) + b2
        print(self.Q_value)
        # 输出是二维数组

    def greedy2(self, state):

        Q_value_out = self.session.run(self.Q_value, feed_dict={self.state_input: [state]})[0]  # 输出的居然是一个【2，1】的向量
        # print(Q_value_out,np.argmax(Q_value_out),) #np.argmax 是最大数索引，不是用来找最大数，只是返回最大数的位置。当然索引位置表示了向左还是向右移动
        self.randomarea -= 0.001 / EXPLORE  # 贪婪策略的时候随机因子的出现频率逐步减少
        x = 0.5 - self.randomarea

        if random.random() < self.randomarea:
            # print("随机了" ,random.randint(0,self.action_dim-1),np.argmax(Q_value_out))
            self.randomtimes += 1
            return random.randint(0, self.action_dim - 1)
        else:
            # print("没有随机", np.argmax(Q_value_out))
            return np.argmax(Q_value_out)



    def greedy(self, state):
        self.epsilon -= (0.5 - 0.01) / 10000
        Q_val_output = self.session.run(self.Q_value, feed_dict={self.state_input: [state]})[0]
        if random.random() <= self.epsilon:
            self.randomtimes += 1
            #print("随机了", self.action_dim,np.argmax(Q_val_output),random.randint(0, self.action_dim - 1) )
            return random.randint(0, self.action_dim - 1)  # 左闭右闭区间，np.random.randint为左闭右开区间

        else:
            #print("没有随机", self.action_dim,np.argmax(Q_val_output))
            return np.argmax(Q_val_output)




    def max_action(self, state):  # 就是为了去掉贪婪策略
        Q_val_output = self.session.run(self.Q_value, feed_dict={self.state_input: [state]})[0]
        action = np.argmax(Q_val_output)
        return action


    def presave(self, state, action, next_state, reword, done):
        now_action = self.action_list[action: action + 1]
        self.memory.append((state, now_action[0], next_state, reword, done))
        if len(self.memory) > self.MEMORY_SIZE:
            self.memory.popleft()
        if len(self.memory) % 50 == 0:
            #print("可以尝试读取记忆思考一次")
            self.traintimes += 1
            # print(self.memory)
            self.train_Q_network()
            #print(self.loss)
            #print("训练一次")




    def create_training_method(self):
        self.action_input = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32)
        self.y_input = tf.placeholder(shape=[None], dtype=tf.float32)  # ???是[None]吗？目的是为了做转置
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)  # 所有action为1 的总数
        self.loss = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0005).minimize(self.loss)



    def save(self):
        self.saver.save(self.session, 'cartpole/model.ckpt')
        print("保存")

    def reload(self):
        self.saver.restore(self.session, 'cartpole/model.ckpt')

    def train_Q_network(self):
        minibatch = random.sample(self.memory, 50)
        state_mini_batch = [data[0] for data in minibatch]
        action_mini_batch = [data[1] for data in minibatch]
        next_state_mini_batch = [data[2] for data in minibatch]
        reword_mini_batch = [data[3] for data in minibatch]
        done_mini_batch = [data[4] for data in minibatch]
        donetimes = 0
        y_batch = []
        Q_next_action = self.session.run(self.Q_value, feed_dict={
            self.state_input: next_state_mini_batch})  # 计算后置的收益，也就是获取下一步期望收益，就是Q函数
        for i in range(0,50):
            # print(done_mini_batch[i])
            if done_mini_batch[i]:
                y_batch.append(reword_mini_batch[i])  # 如果为done的话，则没有对下一步的影响，直接期望值就是 reword
                donetimes += 1
            else:
                y_batch.append(reword_mini_batch[i] +0.9 * np.max(Q_next_action[i]))  # 暂时没有处理好  马尔可夫抉择的 推导过程

        # for i in range(len(minibatch)):
        # print(minibatch[i][0])

        # print("y_batch", y_batch)

        # 接下来要开始进行训练了
        _, self.cost = self.session.run([self.optimizer, self.loss], feed_dict={
            self.y_input: y_batch,
            self.state_input: state_mini_batch,
            self.action_input: action_mini_batch
        })
        #print("坚持了", donetimes, "次")




def main():
    evn = gym.make('CartPole-v0')  # 调用gym的基本的make函数，构造一个环境
    Magent = DQN(evn)  # 创建一个类实体，这时候已经有了W1 W2 B1 B2 了
    #print("A")
    state = evn.reset()  # 拿到初始状态的参数
    # print(state)
    #Magent.reload()
    for i in range(EXPLORE):
        #evn.render()
        state = evn.reset()
        allr = 0
        Magent.randomtimes = 0
        for step in range(2000):
            #evn.render()
            #time.sleep(0.0002)
            action = Magent.greedy(state)  # 拿到了初始状态之后，第一件事就是输入到环境中， Q值直接作为action输入了
            # print("shuchu",action)
            next_stste, reword, done, _ = evn.step(action)
            # 拿到下一步的状态之后，先进行存储，使用perceive函数，记录一个batch之后再开始训练
            Magent.presave(state, action, next_stste, reword, done)
            allr += reword
            state = next_stste  # 更新状态
            if done:
                #print(i,"随机了", Magent.randomtimes, "次，得分", allr, "次")
                break
        if i % 100 == 0:
            total_reward = 0
            for _ in range(50):
                state = evn.reset()
                for _ in range(STEP):
                    evn.render()
                    action = Magent.max_action(state)  # direct action for test
                    state, reward, done, _ = evn.step(action)
                    total_reward += reward
                    #print(total_reward)
                    if done:
                        break
            ave_reward = total_reward / 50
            print('episode: ', i, 'Evaluation Average Reward:', ave_reward)
            print("训练次数",Magent.traintimes,len(Magent.memory))
            #Magent.save()
            Magent.traintimes = 0
            if ave_reward >= 200:
                break

if __name__ == '__main__':
    main()
