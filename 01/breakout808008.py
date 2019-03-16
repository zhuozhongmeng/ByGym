# input all the support

import tensorflow as tf
import numpy as np
import gym as gym
from collections import deque
import random
import cv2
import time as pytime
import matplotlib.pyplot as plt

# set static
GAME = "Breakout-v4"
MEMORYSIZE = 200000  # 保留样本大小
Batch_size = 64  # 训练取样本大小
GAMMA = 0.999999  # 衰减率。伽马值，音译
IMG_WIDTH = 80  # 图像宽度
IMG_HEIGHT = 80  # 图像高度
IMG_TIME_LONG = 8  # 图像时序长度
SHOW_TIMES  = 0
# init Variable 定义及初始化一些全局变量
view_total_reward = []  # 观察总得分分布
view_best_reward = []  # 轮次最高分分布
# -------------------------------------------------------------------------------------------------

# 定义一个图像处理方法，将图像切片变形成（40，40，1）
def ImgProcess(state):
    state1 = state[32:192, 0:160, 0:1]  # 截取有用信息，第一个方法是抽取第一个层图像，等于使用灰度图
    small_state = cv2.resize(state1, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)  # 压缩到需要的画面大小
    # state3 = small_state[:,:,np.newaxis] #最后加一个维度，np.newaxis  新增维度。很字面的意思
    return small_state
    # 这里参考方法进行了图像的处理，调整了图像的曲线。


def show_plt():
    plt.plot(range(len(view_total_reward)),view_total_reward,)
    plt.plot(range(len(view_best_reward)),view_best_reward,)
    plt.savefig("breakout808008/data.png",dpi=1000)
    plt.close()

# -------------------------------------------------------------------------------------------------


class DQN():

    def __init__(self, evn):
        self.action_dim = evn.action_space.n
        self.session = tf.InteractiveSession()
        self.creat_net()  # 一开始就初始化，创建一个网络出来先
        self.memory = deque()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        self.m_reward = 0
        self.insi = 0
        self.training_time = 0
        self.get_action_time = 0
        self.get_action_times = 0
        self.get_memory_time = 0
        self.random_times = 0
        self.m_times = 0
        self.mm_reward = 1
        self.temprandomtimes = 0
        #

    # -------------------------------------------------------------------------------------------------

    def reload(self):
        self.saver.restore(self.session, 'breakout808008/model.ckpt')
        print("读取记忆")

    def save_weight(self):
        self.saver.save(self.session, 'breakout808008/model.ckpt')
        # print("保存成功,样本空间用量",len(self.memory) * 100 / MEMORYSIZE, "%")

    def show_randomtimes(self):
        #print("训练占比",self.m_times / (self.m_times + self.random_times))
        self.temprandomtimes = self.m_times / (self.m_times + self.random_times)
        self.random_times = 0
        self.m_times = 0
        return  self.temprandomtimes

    # -------------------------------------------------------------------------------------------------

    def get_weights(self, shape):
        weight = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(weight)

    def get_bias(self, shape):
        bias = tf.constant(0.01, shape=shape)
        return tf.Variable(bias)


    # -------------------------------------------------------------------------------------------------

    def creat_net(self):  # 创建tensorflow的图，用来直接逼出一个Q的网络价值函数
        self.img_input = tf.placeholder(dtype=tf.float32, shape=[None, IMG_WIDTH, IMG_HEIGHT, IMG_TIME_LONG])
        self.y_input = tf.placeholder(dtype=tf.float32, shape=[None])
        self.action_input = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim])

        w1 = self.get_weights([8, 8, IMG_TIME_LONG, 32])
        b1 = self.get_bias([32])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(self.img_input, w1, [1, 4, 4, 1], padding="SAME") + b1)
        conv1 = tf.nn.max_pool(h_conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
        print("conv1.shape",conv1.shape)
        w2 = self.get_weights([4, 4, 32, 64])
        b2 = self.get_bias([64])
        h_conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w2, [1, 2, 2, 1], padding="SAME") + b2)
        print("h_conv2.shape",h_conv2.shape)
        #h_conv2 = tf.nn.max_pool(h_conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
        #print("h_conv2_after_max_pool.shape", h_conv2.shape)

        w3 = self.get_weights([3, 3, 64, 64])
        b3 = self.get_bias([64])
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, w3, [1, 1, 1, 1], padding="SAME") + b3)
        print("h_conv3.shape", h_conv3.shape)
        w_fc1 = self.get_weights([1600, 512])
        b_fc1 = self.get_bias([512])
        conv3_flat = tf.reshape(h_conv3, [-1,1600])
        print("conv3_flat.shape", conv3_flat.shape)
        h_fc1 = tf.nn.relu(tf.matmul(conv3_flat, w_fc1) + b_fc1)

        w_fc2 = self.get_weights([512, self.action_dim])
        b_fc2 = self.get_bias([self.action_dim])

        self.Q_value = tf.matmul(h_fc1, w_fc2) + b_fc2  # 直到这里，拿到的只是一个图像的识别结果抽象，带action的二维矩阵 当作是价值函数
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)  # 这个是动作价值函数
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))  # y_input 就是最佳策略得分，就是回报，来自于马尔可夫过程结果，这里就是让输出不断的毕竟最佳策略得分

        self.optimizer = tf.train.AdamOptimizer(1e-6).minimize(self.cost)
        print("创建了一个网络")

    # -------------------------------------------------------------------------------------------------

    def training(self):
        minibatch = random.sample(self.memory, Batch_size)  # 这里是利用随机库，在记忆中，随机抽取一定数量minisize = 10 的记忆。然后等待下一步使用。
        mini_state = [data[0] for data in minibatch]
        mini_action = [data[1] for data in minibatch]
        mini_next_state = [data[2] for data in minibatch]
        mini_reward = [data[3] for data in minibatch]
        mini_done = [data[4] for data in minibatch]
        total_Q = []
        next_Q_value = self.Q_value.eval(feed_dict={self.img_input: mini_next_state})

        for i in range(Batch_size):
            if mini_done[i]:
                total_Q.append(mini_reward[i])
                # print("游戏失败")
            else:
                total_Q.append(mini_reward[i] + GAMMA * np.max(next_Q_value[i]))
                # print("记录贝尔曼，",np.argmax(next_Q_value[i]))
                # print(np.shape(total_Q))
        self.optimizer.run(feed_dict={
            self.img_input: mini_state,
            self.action_input: mini_action,
            self.y_input: total_Q
        })
        #writer = tf.summary.FileWriter("log", self.session.graph)

    # -------------------------------------------------------------------------------------------------

    def get_greedy_action(self, state):
        state = np.reshape(state,[1,IMG_WIDTH,IMG_HEIGHT,IMG_TIME_LONG])
        action = self.Q_value.eval(feed_dict={self.img_input: state})
        return action

    def get_action(self, state):
        if self.get_action_times < 99999999:
            self.get_action_times += 1
        random_area = 1 - self.get_action_times * 0.00000001
        if random.random() > random_area/5:
            get_action_time_start = pytime.time()
            action = np.argmax(self.get_greedy_action(state))
            get_action_time_end = pytime.time()
            self.get_action_time += get_action_time_end - get_action_time_start
            self.m_times += 1
        else:
            action = random.randint(0, self.action_dim - 1)
            self.random_times += 1
        return action

    # -------------------------------------------------------------------------------------------------

    def percieve(self, state, action, next_state, reward, done, now_times):

        action_index = np.zeros(self.action_dim)
        action_index[action] = 1
        self.get_memory_time += 1
        self.memory.append([state, action_index, next_state, reward, done, now_times])

        if len(self.memory) > MEMORYSIZE:
            self.memory.popleft()
        if len(self.memory) > Batch_size:
            if self.insi == 0:
                print("试玩结束，开始训练")
                self.insi = 1

            per_training_stare = pytime.time()
            self.training()
            per_training_end = pytime.time()

            self.training_time += per_training_end - per_training_stare


# -------------------------------------------------------------------------------------------------



def main():
    evn = gym.make(GAME)
    agent = DQN(evn)
    #agent.reload()
    init_state = evn.reset()
    init_state = ImgProcess(init_state)
    state_with_times = np.stack((init_state, init_state, init_state, init_state, init_state, init_state, init_state, init_state), axis=2)
    done_times = 0
    round_reward = 0
    best_reward = 0
    round_time_start = pytime.time()

    for times in range(100000000000000):
        evn.render() #是否显示画面
        if times == 0:
            state = state_with_times  # 初始化的时候的state

        action = agent.get_action(state)
        next_state, reward, done, _ = evn.step(action)
        next_state = ImgProcess(next_state)
        next_state = np.reshape(next_state, [IMG_WIDTH, IMG_HEIGHT, 1])

        next_state_with_times = np.append(next_state,state_with_times[:, :, 0:7],  axis=2)  # 记录时序状态
        agent.percieve(state_with_times, action, next_state_with_times, reward, done, times)
        state_with_times = next_state_with_times
        agent.m_reward += reward
        round_reward += reward
        if done:
            if round_reward > best_reward:
                best_reward = round_reward
            done_times += 1
            evn.reset()
            round_reward = 0
            if done_times % 10 == 0:
                round_time_end = pytime.time()
                print( done_times, "局得分：", agent.m_reward, "分，最高单次得分", best_reward, "分，训练比例",agent.show_randomtimes(),"训练用时", agent.training_time, "秒,判断用时", agent.get_action_time, "秒,总用时：",
                      round_time_end - round_time_start, "秒")
                round_time_start = pytime.time()
                agent.training_time = 0
                agent.get_action_time = 0
                view_total_reward.append(agent.m_reward)
                view_best_reward.append(best_reward)
                agent.m_reward = 0
                agent.save_weight()
                best_reward = 0
                show_plt()


if __name__ == '__main__':
    main()
