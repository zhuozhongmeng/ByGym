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
import setting

# set static
GAME = "Breakout-v4"
MEMORYSIZE = 50000  # 保留样本大小
IMG_WIDTH = 160  # 图像宽度
IMG_HEIGHT = 160  # 图像高度
IMG_DEPTH = 1 #图像深度
IMG_TIME_LONG = 8  # 图像时序长度
OBSEVER_TIMES = 100 #一开始随便玩的次数
TIMES_PER_ROUNDS = 3500 #限制每局最高动作数
totalreward = 0
# init Variable 定义及初始化一些全局变量
view_total_reward = []  # 观察总得分分布

view_best_reward = []  # 轮次最高分分布
times_list=[]   #每局动作次数分布
view_total_reward = setting.load_data(filename="view_total_reward.txt")
view_best_reward = setting.load_data(filename="view_best_reward.txt")
Temp_getaction = int(setting.load_single_data(filename="data.txt"))
# -------------------------------------------------------------------------------------------------

# 定义一个图像处理方法，将图像切片变形成（40，40，1）
def ImgProcess300300(state):
    state = state[32:192, 0:160, 0:1]  # 截取有用信息，第一个方法是抽取第一个层图像，等于使用灰度图
    #small_state = cv2.resize(state1, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)  # 压缩到需要的画面大小
    state = state.reshape([160,160]) #最后加一个维度，np.newaxis  新增维度。很字面的意思
    return state
    # 这里参考方法进行了图像的处理，调整了图像的曲线。


#def ColorMat2Binary(state):
#    height = state.shape[0]
#    width = state.shape[1]
#    nchannel = state.shape[2]
#    sHeight = int(height * 0.5)
#    sWidth = IMG_WIDTH
#
#    state_gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
#
#    _, state_binary = cv2.threshold(state_gray, 5, 255, cv2.THRESH_BINARY)
#
#    state_binarySmall = cv2.resize(state_binary, (sWidth, sHeight), interpolation=cv2.INTER_AREA)
#
#    cnn_inputImg = state_binarySmall[25:, :]
#    cnn_inputImg = cnn_inputImg.reshape((IMG_WIDTH, IMG_HEIGHT))

#    return cnn_inputImg



def show_plt():
    plt.plot(range(len(view_total_reward)),view_total_reward,'.')
    plt.savefig("breakout8080deepconv/summary_10round.png", dpi=500)
    plt.close()
    plt.plot(range(len(view_best_reward)),view_best_reward,'.')
    plt.savefig("breakout8080deepconv/summary_best.png",dpi=500)
    plt.close()
    plt.plot(range(len(times_list)), times_list, '.')
    plt.savefig("breakout8080deepconv/summary_timeslist.png", dpi=500)
    plt.close()

# -------------------------------------------------------------------------------------------------


class DQN():

    def __init__(self, evn,istrain,temp_getaction=0):
        self.action_dim = evn.action_space.n
        self.session = tf.InteractiveSession()
        self.creat_net()  # 一开始就初始化，创建一个网络出来先
        self.memory = deque()
        print("istrain",istrain)
        self.istrain = istrain
        self.merge = tf.summary.merge_all()
        self.sum_writer = tf.summary.FileWriter("breakout8080deepconv/",self.session.graph)
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
        self.getaction = int(temp_getaction)
        self.nowreward = 0


    # -------------------------------------------------------------------------------------------------

    def reload(self):
        self.saver.restore(self.session, 'breakout8080deepconv/model.ckpt')
        print("读取记忆")

    def save_weight(self):
        self.saver.save(self.session, 'breakout8080deepconv/model.ckpt')
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
    def training(self,Batch_size=32,GAMMA=0.99):
        #print("训练中的batch是",Batch_size)
        Batch_size=int(Batch_size)
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
        if self.getaction % 10 ==0:
            thesummary = self.session.run(self.merge,feed_dict={ self.img_input: mini_state,self.action_input: mini_action,self.y_input: total_Q })
            self.sum_writer.add_summary(thesummary,self.getaction)
    # -------------------------------------------------------------------------------------------------

    def creat_net(self):  # 创建tensorflow的图，用来直接逼出一个Q的网络价值函数
        with tf.name_scope("input_layer"):
            self.img_input = tf.placeholder(dtype=tf.float32, shape=[None, IMG_WIDTH, IMG_HEIGHT, IMG_TIME_LONG])
            self.y_input = tf.placeholder(dtype=tf.float32, shape=[None])
            self.action_input = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim])
        with tf.name_scope("layer1"):
            print("input_image.shape",self.img_input.shape)

            layer1_w = self.get_weights([8, 8, 8, 32])
            self.summary_layer1_w = tf.summary.histogram("layer1_w",layer1_w)
            print("layer1_w1.shape",layer1_w.shape)

            layer1_b = self.get_bias([32])
            self.summary_layer1_b = tf.summary.histogram('layer1_b',layer1_b)
            print("layer1_b1.shape", layer1_b.shape)

            layer1_conv = tf.nn.relu(tf.nn.conv2d(self.img_input, layer1_w, [1, 2, 2, 1], padding="SAME") + layer1_b,name="layer1_conv")
            img_layer1_conv = layer1_conv[ : , : ,: , 0:3 ]
            self.summary_layer1_conv = tf.summary.image("layer1_conv",img_layer1_conv,max_outputs=16)
            print("layer1_conv.shape",layer1_conv.shape)

            #以下是暂时不用的池化层
            #conv1 = tf.nn.max_pool(h_conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
            #self.m_conv1 = tf.summary.histogram("m_conv1", conv1)
            #print("conv1.shape",conv1.shape)


        with tf.name_scope("layer2"):
            layer2_w = self.get_weights([8, 8, 32, 64])
            self.summary_w2 =  tf.summary.histogram("layer2_w",layer2_w)
            print("layer2_w.shape",layer2_w.shape)

            layer2_b = self.get_bias([64])
            self.summary_b2 = tf.summary.histogram("layer2_b",layer2_b)
            print("layer2_b.shape",layer2_b.shape)

            layer2_conv = tf.nn.relu(tf.nn.conv2d(layer1_conv, layer2_w, [1, 2, 2, 1], padding="SAME") + layer2_b)
            img_layer2_conv = layer2_conv[ : , : ,: , 0:3 ]
            self.summary_layer2_conv = tf.summary.image("layer2_conv",img_layer2_conv,max_outputs=16)
            print("layer2_conv.shape",layer2_conv.shape)
            #h_conv2 = tf.nn.max_pool(layer2_conv, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
            #print("h_conv2_after_max_pool.shape", h_conv2.shape)
        with tf.name_scope("layer3"):
            layer3_w = self.get_weights([8, 8, 64, 128])
            self.summary_layer3_w = tf.summary.histogram("layer3_w",layer3_w)
            print("layer3_w.shape",layer3_w.shape)

            layer3_b = self.get_bias([128])
            self.summary_layer3_b = tf.summary.histogram("layer3_b",layer3_b)
            print("layer3_b.shape",layer3_b.shape)

            layer3_conv = tf.nn.relu(tf.nn.conv2d(layer2_conv, layer3_w, [1, 2, 2, 1], padding="SAME") + layer3_b)
            img_layer3_conv = layer3_conv[:, :, :, 0:3]
            self.summary_layer3_conv = tf.summary.image("layer3_conv", img_layer3_conv, max_outputs=16)
            print("layer3_conv.shape", layer3_conv.shape)


        with tf.name_scope("layer4"):
            layer4_w = self.get_weights([4,4,128,256])
            self.summary_layer4_w = tf.summary.histogram("layer4_w",layer4_w)
            print("layer4_w.shape",layer4_w.shape)

            layer4_b = self.get_bias([256])
            self.summary_layer4_b = tf.summary.histogram("layer4_b",layer4_b)
            print("layer4_b.shape",layer4_b.shape)

            layer4_conv = tf.nn.relu(tf.nn.conv2d(layer3_conv,layer4_w,[1,2,2,1],padding="SAME")+layer4_b)
            img_layer4_conv = layer4_conv[:, :, :, 0:3]
            self.summary_layer4_conv = tf.summary.image("layer4_conv", img_layer4_conv, max_outputs=16)
            print("layer4_conv.shape",layer4_conv.shape)

        with tf.name_scope("layer5"):
            layer5_w = self.get_weights([4,4,256,256])
            self.summary_layer5_w = tf.summary.histogram("layer5_w",layer5_w)
            print("layer5_w.shape",layer5_w.shape)

            layer5_b = self.get_bias([256])
            self.summary_layer5_b = tf.summary.histogram("layer5_b",layer5_b)
            print("layer5_b.shape",layer5_b.shape)

            layer5_conv = tf.nn.relu(tf.nn.conv2d(layer4_conv,layer5_w,[1,2,2,1],padding="SAME")+layer5_b)
            img_layer5_conv = layer5_conv[:, :, :, 0:3]
            self.summary_layer5_conv = tf.summary.image("layer5_conv", img_layer5_conv, max_outputs=16)
            print("layer5_conv.shape",layer5_conv.shape)


        with tf.name_scope("layer_fullconnet1"):
            w_fc1 = self.get_weights([6400, 512])
            b_fc1 = self.get_bias([512])
            conv3_flat = tf.reshape(layer5_conv, [-1,6400])
            print("conv3_flat.shape", conv3_flat.shape)
            h_fc1 = tf.nn.relu(tf.matmul(conv3_flat, w_fc1) + b_fc1)

            w_fc2 = self.get_weights([512, self.action_dim])
            b_fc2 = self.get_bias([self.action_dim])
            #with tf.name_scope("output"):
        with tf.name_scope("output_layer"):
            self.Q_value = tf.matmul(h_fc1, w_fc2) + b_fc2  # 直到这里，拿到的只是一个图像的识别结果抽象，带action的二维矩阵 当作是价值函数
            Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)  # 这个是动作价值函数
            self.summary_Qvalue = tf.summary.histogram("Summary_Qvalue",self.Q_value)
            self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))  # y_input 就是最佳策略得分，就是回报，来自于马尔可夫过程结果，这里就是让输出不断的毕竟最佳策略得分
            self.summary_cost = tf.summary.scalar("Summary_cost",self.cost)
        self.optimizer = tf.train.AdamOptimizer(1e-6).minimize(self.cost)
        print("构建实体图完成")
    # -------------------------------------------------------------------------------------------------

    def get_greedy_action(self, state):
        #state = np.reshape(state,[1,IMG_WIDTH,IMG_HEIGHT,IMG_TIME_LONG])
        action = self.Q_value.eval(feed_dict={self.img_input: [state]})[0]
        #print(action,np.argmax(action))
        argmaxaction = np.argmax(action)
        #print("no[0]",self.Q_value.eval(feed_dict={self.img_input: state}),np.argmax(self.Q_value.eval(feed_dict={self.img_input: state})))
        return np.argmax(action)

    def get_action(self, state):
        self.getaction += 1
        if self.getaction > 10000:
            random_action = 10000
        else:
            random_action = self.getaction
        #if self.get_action_times < 999999999:
        #    self.get_action_times += 1
       # random_area = 1 - self.get_action_times * 0.000000001
        if random.random() > 1 - (random_action + self.nowreward /100) /10011:
            get_action_time_start = pytime.time()
            action = self.get_greedy_action(state)
            get_action_time_end = pytime.time()
            self.get_action_time += get_action_time_end - get_action_time_start
            self.m_times += 1
        else:
            action = random.randint(0, self.action_dim - 1)
            self.random_times += 1
        return action

    # -------------------------------------------------------------------------------------------------

    def percieve(self, state, action, next_state, reward, done, now_times,Batch_size,Is_train = 1,GAMMA=0.99):
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
            if Is_train == 1:
                self.training(Batch_size=Batch_size,GAMMA=GAMMA)
                #print("训练gamma=",GAMMA)
            else :
                print("不训练")
            per_training_end = pytime.time()

            self.training_time += per_training_end - per_training_stare


# -------------------------------------------------------------------------------------------------



def main():
    print(datetime.datetime.now())
    evn = gym.make(GAME)
    agent = DQN(evn,istrain = 1,temp_getaction=Temp_getaction)
    print("测试istrain",agent.istrain,"获取历史总判断数",agent.getaction)
    print(evn.action_space)
    agent.reload()
    round_reward = 0
    best_reward = 0
    round_10_reward = 0
    round_time_start = pytime.time()  # --------------------------------------------获取本局开始时间
    state_with_4times = None
    state_with_4times = None

    for rounds in range(100000000000000):
        #每一局都读取一次整套基础设置参数
        istrain, gamma, batchsize, isrender, timesperround, roundpershow = setting.load_setting(filename="setting.txt")
        #print("读取出来的", istrain, gamma, batchsize, isrender, timesperround, roundpershow)
        #istrain收否训练, gamma衰减值， batchsize单词训练样本容量, isrender是否显示图像, timesperround每局最多行动数, roundpershow每几次打印一次图像
        state = evn.reset()
        state = ImgProcess300300(state)
        state_with_4times = np.stack((state, state, state, state, state, state, state, state), axis=2)
        for times in range(TIMES_PER_ROUNDS):
            #print("start",times)
            evn.render() #是否显示画面
            action = agent.get_action(state_with_4times)
            #print("action",action,agent.get_action(state_with_4times))
            next_state, reward, done, _ = evn.step(action)
            #print(next_state, reward, done, _ )
            next_state = ImgProcess300300(next_state)
            next_state = np.reshape(next_state, [IMG_WIDTH, IMG_HEIGHT,1])
            next_state_with_4times = np.append(next_state,state_with_4times[:, :, :7],  axis=2)  # 记录时序状态
            if istrain == 1:
                agent.percieve(state_with_4times, action, next_state_with_4times, reward, done, times,Batch_size=batchsize,Is_train=istrain,GAMMA=gamma)
            state_with_4times = next_state_with_4times #更新输入状态
            round_reward += reward
            round_10_reward += reward


            if done:
                break

        if round_reward > best_reward:
            best_reward = round_reward
        #print(done)
        print(round_reward,'[',times,']')
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
            setting.save_data(data=view_total_reward,filename="view_total_reward.txt")
            setting.save_data(data=view_best_reward,filename="view_best_reward.txt")
            setting.save_signle_data(data=float(agent.getaction),filename="data.txt")
            print("当前判断总次数",str(agent.getaction))

if __name__ == '__main__':
    main()
