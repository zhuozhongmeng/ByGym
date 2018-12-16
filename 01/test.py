import  gym
import time

env = gym.make("CartPole-v0")

env.reset()
action = 0
for q in range(20):
    env.reset()
    print(env.observation_space,env.action_space)
    for i in range(1000):
        env.render()

        time.sleep(0.001)

        a,b,c,d = env.step(action)
        if a[1] >= 0:
            action = 0
        if a[1] < 0 :
            action = 1

        print(a,b,c,d)

        if c :
            print("失败了")
            break