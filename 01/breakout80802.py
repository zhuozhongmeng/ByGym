# input all the support

import tensorflow as tf
import numpy as np
import gym as gym
from collections import deque
import random
import cv2
import time as pytime
import matplotlib.pyplot as plt
import datetime
# set static
GAME = "Breakout-v4"
MEMORYSIZE = 50000  # 保留样本大小
Batch_size = 32  # 训练取样本大小
GAMMA = 0.99  # 衰减率。伽马值，音译
IMG_WIDTH = 80  # 图像宽度
IMG_HEIGHT = 80  # 图像高度
IMG_DEPTH = 1 #图像深度
IMG_TIME_LONG = 4  # 图像时序长度
INI_EPSILON = 1 #初始随机探索比例
FINAL_EPSILON = 0.0001 #最终随机探索比例
OBSEVER_TIMES = 500 #一开始随便玩的次数
TIMES_PER_ROUNDS = 1500 #限制每局最高动作数
totalreward = 0
# init Variable 定义及初始化一些全局变量
view_total_reward = []  # 观察总得分分布
view_best_reward = []  # 轮次最高分分布
times_list=[]   #每局动作次数分布
CNN_INPUT_WIDTH = 80
CNN_INPUT_HEIGHT = 80
CNN_INPUT_DEPTH = 1
SERIES_LENGTH = 4
# -------------------------------------------------------------------------------------------------

# 定义一个图像处理方法，将图像切片变形成（40，40，1）
def ColorMat2B(self, state):  ##used for the game flappy bird
    height = 80
    width = 80
    state_gray = cv2.cvtColor(cv2.resize(state, (height, width)), cv2.COLOR_BGR2GRAY)
    _, state_binary = cv2.threshold(state_gray, 5, 255, cv2.THRESH_BINARY)
    state_binarySmall = cv2.resize(state_binary, (width, height))
    cnn_inputImage = state_binarySmall.reshapeh((height, width))
    return cnn_inputImage


def ColorMat2Binary(state):
    # state_output = tf.image.rgb_to_grayscale(state_input)
    # state_output = tf.image.crop_to_bounding_box(state_output,34,0,160,160)
    # state_output = tf.image.resize_images(state_output,80,80,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
    # state_output = tf.squeeze(state_output)
    # return state_output

    height = state.shape[0]
    width = state.shape[1]
    nchannel = state.shape[2]

    sHeight = int(height * 0.5)
    sWidth = CNN_INPUT_WIDTH

    state_gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    # print state_gray.shape
    # cv2.imshow('test2',state_gray)
    # cv2.waitkey(0)

    _, state_binary = cv2.threshold(state_gray, 5, 255, cv2.THRESH_BINARY)

    state_binarySmall = cv2.resize(state_binary, (sWidth, sHeight), interpolation=cv2.INTER_AREA)

    cnn_inputImg = state_binarySmall[25:, :]
    # rstArray = state_graySmall.reshape(swidth * sHeight )
    cnn_inputImg = cnn_inputImg.reshape((CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT))

    return cnn_inputImg


def show_plt():
    plt.plot(range(len(view_total_reward)),view_total_reward,'.')
    plt.savefig("breakout8080/10round1.png", dpi=1000)
    plt.close()
    plt.plot(range(len(view_best_reward)),view_best_reward,'.')
    plt.savefig("breakout8080/best1.png",dpi=1000)
    plt.close()
    plt.plot(range(len(times_list)), times_list, '.')
    plt.savefig("breakout8080/timeslist1.png", dpi=1000)
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
        self.getaction = 10000
        self.nowreward = 0

    # -------------------------------------------------------------------------------------------------

    def reload(self):
        self.saver.restore(self.session, 'breakout8080/model.ckpt')
        print("读取记忆")

    def save_weight(self):
        self.saver.save(self.session, 'breakout8080/model.ckpt')
        #print("保存成功,样本空间用量",len(self.memory) * 100 / MEMORYSIZE, "%")

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
        INPUT_DEPTH = SERIES_LENGTH
        self.input_layer = tf.placeholder(tf.float32, [None, CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT, INPUT_DEPTH],
                                          name='status-input')
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])
        self.y_input = tf.placeholder(tf.float32, [None])

        W1 = self.get_weights([8, 8, 4, 32])
        b1 = self.get_bias([32])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(self.input_layer, W1, strides=[1, 4, 4, 1], padding='SAME') + b1)
        conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W2 = self.get_weights([4, 4, 32, 64])
        b2 = self.get_bias([64])
        h_conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides=[1, 2, 2, 1], padding='SAME') + b2)
        # conv2 = tf.nn.max_pool( h_conv2, ksize = [ 1, 2, 2, 1 ], strides= [ 1, 2, 2, 1 ], padding= 'SAME' )

        W3 = self.get_weights([3, 3, 64, 64])
        b3 = self.get_bias([64])
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3)

        W_fc1 = self.get_weights([1600, 512])
        b_fc1 = self.get_bias([512])
        # h_conv2_flat = tf.reshape( h_conv2, [ -1, 11 * 11 * 32 ] )

        conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        h_fc1 = tf.nn.relu(tf.matmul(conv3_flat, W_fc1) + b_fc1)

        W_fc2 = self.get_weights([512, self.action_dim])
        b_fc2 = self.get_bias([self.action_dim])
        self.Q_value = tf.matmul(h_fc1, W_fc2) + b_fc2

        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,
                                             self.action_input), reduction_indices=1)

        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))

        self.optimizer = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    # -------------------------------------------------------------------------------------------------

    def training(self):
        minibatch = random.sample(self.memory, Batch_size)  # 这里是利用随机库，在记忆中，随机抽取一定数量minisize = 10 的记忆。然后等待下一步使用。
        mini_state = [data[0] for data in minibatch]
        mini_action = [data[1] for data in minibatch]
        mini_next_state = [data[2] for data in minibatch]
        mini_reward = [data[3] for data in minibatch]
        mini_done = [data[4] for data in minibatch]
        total_Q = []
        next_Q_value = self.Q_value.eval(feed_dict={self.input_layer: mini_next_state})
        print(next_Q_value)
        for i in range(Batch_size):
            if mini_done[i]:
                total_Q.append(mini_reward[i])
                # print("游戏失败")
            else:
                total_Q.append(mini_reward[i] + GAMMA * np.max(next_Q_value[i]))
                # print("记录贝尔曼，",np.argmax(next_Q_value[i]))
                # print(np.shape(total_Q))
        self.optimizer.run(feed_dict={
            self.input_layer: mini_state,
            self.action_input: mini_action,
            self.y_input: total_Q
        })
        #writer = tf.summary.FileWriter("log", self.session.graph)

    # -------------------------------------------------------------------------------------------------

    def get_greedy_action(self, state):
        #state = np.reshape(state,[1,IMG_WIDTH,IMG_HEIGHT,IMG_TIME_LONG])
        action = self.Q_value.eval(feed_dict={self.input_layer: [state]})[0]
        print(action,np.argmax(action))
        #print("no[0]",self.Q_value.eval(feed_dict={self.img_input: state}),np.argmax(self.Q_value.eval(feed_dict={self.img_input: state})))
        return np.argmax(action)

    def get_action(self, state):
        self.getaction += 1
        if self.getaction > 10000:
            self.getaction = 10000
        #if self.get_action_times < 999999999:
        #    self.get_action_times += 1
       # random_area = 1 - self.get_action_times * 0.000000001
        if random.random() > 1 - (self.getaction + self.nowreward /100) /10031:
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
        if len(self.memory) > OBSEVER_TIMES:
            if self.insi == 0:
                print("试玩结束，开始训练")
                self.insi = 1

            per_training_stare = pytime.time()
            self.training()
            per_training_end = pytime.time()

            self.training_time += per_training_end - per_training_stare


# -------------------------------------------------------------------------------------------------



def main():
    print(datetime.datetime.now())
    evn = gym.make(GAME)
    agent = DQN(evn)
    print(evn.action_space)
    agent.reload()
    round_reward = 0
    best_reward = 0
    round_10_reward = 0
    round_time_start = pytime.time()  # --------------------------------------------获取本局开始时间
    state_with_4times = None
    next_state_with_4times = None

    for rounds in range(100000000000000):

        state = evn.reset()
        #print("reset",datetime.datetime.now())
        state = ColorMat2Binary(state)
        state_with_4times = np.stack((state, state, state, state), axis=2)
        for times in range(TIMES_PER_ROUNDS):
            #print("start",times)
            evn.render() #是否显示画面
            action = agent.get_action(state_with_4times)
            next_state, reward, done, _ = evn.step(action)
            next_state = np.reshape(ColorMat2Binary(next_state),(80,80,1))
            next_state_with_4times = np.append(next_state, state_with_4times[:, :, :3], axis=2)

            agent.percieve(state_with_4times, action, next_state_with_4times, reward, done, times)

            state_with_4times = next_state_with_4times #更新输入状态
            round_reward += reward
            round_10_reward += reward


            if done:
                break

        if round_reward > best_reward:
            best_reward = round_reward
        print(done)
        print(rounds,"局得分", round_reward,'|',round_10_reward,"分|有",times,'次动作',datetime.datetime.now())
        times_list.append(times)
        round_reward = 0

        if rounds % 10 == 0:
            round_time_end = pytime.time()  # ------------------------------------------获取本局结束时间
            print(rounds, "局得分：", round_10_reward, "分，最高", best_reward, "分，训练",
                  agent.show_randomtimes(), "，用时", agent.training_time, "秒,判断用", agent.get_action_time, "秒,总用：",
                  round_time_end - round_time_start, "秒,memoryusing",len(agent.memory),datetime.datetime.now())
            view_total_reward.append(round_10_reward)
            view_best_reward.append(best_reward)
            agent.nowreward = round_10_reward
            round_10_reward = 0
            agent.save_weight()
            best_reward = 0
            show_plt()
            agent.get_action_time = 0
            agent.training_time = 0
            round_time_start = pytime.time()  # --------------------------------------------获取本局开始时间



if __name__ == '__main__':
    main()
