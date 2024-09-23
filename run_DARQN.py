from env_CD import *
from tqdm import tqdm
import random
import torch
import torch.optim as optim
from src.model_DARQN import DARQN
from src.memory import Memory_DARQN as Memory
# from tensorboardX import SummaryWriter
from collections import deque
import matplotlib.pyplot as plt
import gc
from src.config_DARQN import initial_exploration, batch_size, update_target,epsilon, grid_num, device, replay_memory_capacity, lr,feature_num, sequence_length,action_space,human_flag,intervention_mode

import os
import pandas as pd
from Qlearning import QLearningTable

import tkinter as tk
from tkinter import filedialog
from PIL import Image
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import collections
import matplotlib
matplotlib.use('Agg')



def update_target_model(online_net, target_net):
    '''
    Synchronize target network parameters
    :param online_net:
    :param target_net:
    :return:
    '''
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())

def select_action(state,target_net,epsilon,action_space,hidden):
    '''
     Select Action
     :param state: 
     :param target_net: 
     :param epsilon: 
     :param action_space: 
     :return:
     '''

    action, hidden = target_net.get_action(state, hidden)
    if np.random.rand() <= epsilon:
        return random.choice(action_space), hidden
    else:
        return action_space[action], hidden

def supervisor(env,divide_flag,QLearning):
    '''
    The supervisor is a Q-learning network with a status of [number of workers, number of tasks, whether divided], an action of [intervention, no intervention], and a reward of [-1 (if the actions of the person and decision-maker are consistent, it indicates that intervention is ineffective), and 1 (if the actions of the person and decision-maker are inconsistent, it indicates that intervention is effective)
    :param env: 
    :param divide_flag:
    :param QLearning: Q-learning model
    :return:
    '''

    worker_num=len(env.bufferPool.worker_list)
    task_num=len(env.bufferPool.task_list)
    state=[worker_num,task_num,divide_flag]
    action=QLearning.choose_action(state)

    return state,action

human_decion=0

def confirm(win,env):
    '''
    Button functions in human-computer interaction interface
    :param win: 
    :param env: 
    :return:
    '''
    global human_decion
    #Agree to agent decision-making
    human_decion=1
    env.human_confirm_num+=1
    win.quit()
    # time.sleep(1)

def reject(win,env):
    '''
    Button functions in human-computer interaction interface
    :param win:
    :return:
    '''
    global human_decion
    #Reject intelligent agent decisions
    human_decion = 0
    env.human_reject_num += 1
    win.quit()
    # time.sleep(1)

def human_decision(env,divide_flag,current_time):
    '''
    Human make final decisions based on the environment, with a comprehensive information of the spatiotemporal distribution of workers and tasks as the state, which is an intuitive decision
    :param env:
    :return:
    '''

    human_action=0
    worker_list = env.bufferPool.worker_list
    task_list = env.bufferPool.task_list

    win = tk.Tk()
    win.title('Human Machine Interface')
    win.geometry('750x650')
    win.resizable(0, 0)

    figure = plt.figure(figsize=(7, 6))  


    ax = figure.add_subplot(1, 1, 1)  

    x_worker=[]
    y_worker=[]
    txt_worker=[]
    for worker in worker_list:
        x_worker.append(float(worker.lng))
        y_worker.append(float(worker.lat))
        txt_worker.append(worker.capacity-len(worker.schedule))

    x_task=[]
    y_task=[]
    txt_task=[]
    for task in task_list:
        x_task.append(float(task.lng))
        y_task.append(float(task.lat))
        txt_task.append(task.deadline-current_time)


    ax.scatter(x_worker, y_worker, s=10, c='r', marker="o", alpha=1, lw=2, label='Worker')  
    ax.scatter(x_task, y_task, s=10, c='g', marker="^", alpha=1, lw=2, label='Task')  
    ax.set_title("Current Temporal and Spatial Distribution", color='black', fontsize=12)
    ax.legend(loc=2, borderaxespad=2, bbox_to_anchor=(0.25, 0.0), ncol=2)
    for i in range(len(x_worker)):
        ax.annotate(txt_worker[i], xy=(x_worker[i], y_worker[i]), color='red', fontsize=7)
    for i in range(len(x_task)):
        ax.annotate(txt_task[i], xy=(x_task[i], y_task[i]), color='green', fontsize=7)

    canvas = FigureCanvasTkAgg(figure, win)
    canvas.draw()
    canvas.get_tk_widget().pack()

    ttk.Separator(win, orient='horizontal').pack(fill=tk.X)
   
    if divide_flag==1:
        decision_text = 'Partition'
    else:
        decision_text = 'No Partition'
    tk.Label(win, text="Agent action: ", font=('Times New Roman', 20, "italic")).place(x=10, y=10)
    tk.Label(win, text=decision_text, font=('Times New Roman', 20, "bold"), fg='blue').place(x=130, y=10)
    tk.Label(win, text="Decision number: ", font=('Times New Roman', 16, "bold")).place(x=510, y=12)
    tk.Label(win, text=env.human_decision_num, font=('Times New Roman', 16, 'bold')).place(x=640, y=12)
    tk.Label(win, text="(", font=('Times New Roman', 16, 'bold')).place(x=670, y=12)
    tk.Label(win, text=env.human_confirm_num, font=('Times New Roman', 16, 'bold'), fg="#1D8348").place(x=680, y=12)
    tk.Label(win, text="/", font=('Times New Roman', 16, 'bold')).place(x=700, y=12)
    tk.Label(win, text=env.human_reject_num, font=('Times New Roman', 16, 'bold'), fg="#CB4335").place(x=715, y=12)
    tk.Label(win, text=")", font=('Times New Roman', 16, 'bold')).place(x=730, y=12)
    
    tk.Button(text='Confirm', command=lambda: confirm(win,env), activeforeground='#1E88E5').place(x=200, y=610, width=70, height=30)
    tk.Button(text='Reject', command=lambda: reject(win,env), activeforeground='#F44336').place(x=470, y=610, width=70, height=30)

 
    s_time=time.time()
    
    e_time = time.time()
    time.sleep(0.1)
    win.destroy()
    plt.close('all')
    env.human_decision_num+=1

    global human_decion
    human_action=human_decion

    return human_action,e_time-s_time

def debug_memory():
    '''
    Video memory debugging
    :return:
    '''
    # for o in gc.get_objects():
    #     if torch.is_tensor(o):
    #         print(o)
    # print("maxrss={}".format(resources.getrusage(resources.RUSAGE_SELF).ru_maxrss))
    tensors=collections.Counter((str(o.device),o.dtype,tuple(o.shape),str(o.data))
                                for o in gc.get_objects()
                                if torch.is_tensor(o))

    for line in sorted(tensors.items()):

        print('{}\t{}'.format(*line))

def Simulation(env,epoch_num,online_net,target_net,optimizer,memory,loss_list,raward_list,q_online_list,q_target_list,QLearning,completion_rate_list):
    '''
     
    :param env: 
    :param epoch_num:
    :param online_net:Q-network
    :param target_net:target network
    :param optimizer:
    :param memory:
    :param loss_list: loss
    :param reward_list: reward 
    :return:
    '''
    epsilon_factor=epsilon

    hidden = None

    for epoch in range(epoch_num):
        rl_log_file = "./results/log/CD_epoch_" + str(epoch) + "_mode_" + str(intervention_mode)+ "_DARQN_rl_log.txt"
        human_log_file = "./results/log/CD_epoch_" + str(epoch) + "_mode_" + str(intervention_mode)+ "_DARQN_human_log.txt"

        rl_log_list = []
        human_log_list = []
        human_time_list = []

        s_time = time.time()
        print("epoch:%s/%s" % (epoch,epoch_num))
        window_time_slice = 0

        state = env.get_env_state(env.ORIGINAL_TIME, window_time_slice)
        state = torch.Tensor(state).to(device)

        divide_flag=0
        train_step=0

        for current_time_step in tqdm(range(env.ORIGINAL_TIME, env.ORIGINAL_TIME+86400)):
            if current_time_step % env.time_slice==0:
                intervention_flag = None
                state_window = None
                reward_window = None
                human_action = None

                if window_time_slice<action_space[0]:
                    divide_flag=0
                    action, hidden = select_action(state, target_net, epsilon_factor, action_space,hidden)
                elif window_time_slice>=action_space[-1]:
                    divide_flag = 1
                    action=action_space[-1]
                else:
                    action, hidden=select_action(state,target_net,epsilon_factor,action_space,hidden)
                  
                    if action >= window_time_slice - 1 and action <= window_time_slice + 1:
                        divide_flag = 1
                    else:
                        divide_flag = 0
                    #Intervention mode [0: Division 1 only: Division 2 only: Full time intervention]
                    if intervention_mode == 0:
                        if divide_flag == 1:
                            if human_flag:
                                # Supervisor, determine if human intervention is needed
                                state_window, intervention_flag = supervisor(env, divide_flag, QLearning)
                                if intervention_flag:
                                    # If the supervisor believes that human intervention is necessary, then the person will make the final decision
                                    human_action,human_time = human_decision(env, divide_flag, current_time_step)
                                    # Record the supervisor's log, including each decision intention (agree or refuse) and time.
                                    human_log_list.append(str(human_action) + ',' + str(human_time))
                                    human_time_list.append(human_time)
                                    # Supervision reward, if a person intervenes, it is effective at 1 and ineffective at -1
                                    if divide_flag == human_action:
                                        reward_window = -1
                                    else:
                                        reward_window = 1

                                    divide_flag = human_action
                                else:
                                    reward_window = 0
                    elif intervention_mode == 1:
                        if divide_flag == 0:
                            if human_flag:
                                state_window, intervention_flag = supervisor(env, divide_flag, QLearning)
                                if intervention_flag:
                                    
                                    human_action,human_time = human_decision(env, divide_flag, current_time_step)
                                    human_log_list.append(str(human_action) + ',' + str(human_time))
                                    human_time_list.append(human_time)
                                    if divide_flag == human_action:
                                        reward_window = -1
                                    else:
                                        reward_window = 1

                                    divide_flag = human_action
                                else:
                                    reward_window = 0
                    elif intervention_mode == 2:
                        if human_flag:
                            state_window, intervention_flag = supervisor(env, divide_flag, QLearning)
                            if intervention_flag:
                                human_action,human_time = human_decision(env, divide_flag, current_time_step)
                                
                                human_log_list.append(str(human_action) + ',' + str(human_time))
                                human_time_list.append(human_time)
                                if divide_flag == human_action:
                                    reward_window = -1
                                else:
                                    reward_window = 1

                                divide_flag = human_action
                            else:
                                reward_window = 0

                reward,next_state=env.work_thread(divide_flag,current_time_step,window_time_slice)

                next_state = torch.Tensor(next_state).to(device)
                env.total_reward+=reward
                # raward_list.append(env.total_reward)

                rl_log_list.append(env.total_reward)

                human=0
                action_one_hot = np.zeros(len(action_space))
                action_one_hot[action_space.index(action)] = 1
                memory.push(state, next_state, action_one_hot, reward,divide_flag)
                state=next_state
                train_step += 1
                if divide_flag==0:
                    window_time_slice+=1
                else:
                    window_time_slice=0
            # Q-learning learn
            if intervention_flag != None:
                worker_num = len(env.bufferPool.worker_list)
                task_num = len(env.bufferPool.task_list)
                next_state_window = [worker_num, task_num, human_action]
                QLearning.learn(state_window, intervention_flag, reward_window, next_state_window)


            if train_step > initial_exploration:
                epsilon_factor -= 0.000005
                epsilon_factor= max(epsilon_factor, 0.01)

                batch = memory.sample(batch_size)
                loss,q_online,q_target = DARQN.train_model(online_net, target_net, optimizer, batch)
                # loss=loss.detach().cpu().numpy().tolist()

                env.total_loss += loss.detach().item()
                env.q_online += q_online.detach().item()
                env.q_target += q_target.detach().item()
                # env.total_loss += loss.detach().cpu().numpy().tolist()
                # env.q_online += q_online.detach().cpu().numpy().tolist()
                # env.q_target += q_target.detach().cpu().numpy().tolist()

                # print(loss)
                # loss_list.append(loss)

                del batch, loss, q_online, q_target

                if train_step % update_target == 0:
                    update_target_model(online_net, target_net)


        e_time = time.time()

        print("This round of simulation cycle takes time:",e_time-s_time)
        print("Total task num:",env.total_tasks_num)
        print("Comlete task num:", env.complete_tasks_num)
        print("Total reward:", env.total_reward)
        print("Total loss:", env.total_loss)
        print("Total q online:", env.q_online)
        print("Total q target :", env.q_target)

        rl_log_f = open(rl_log_file, "w")
        for i in range(len(rl_log_list)):
            rl_log = rl_log_list[i]
            rl_log_f.write(str(i) + ',' + str(rl_log) + '\n')
        rl_log_f.close()

        human_log_f = open(human_log_file, "w")
        for i in range(len(human_log_list)):
            human_log = human_log_list[i]
            human_log_f.write(str(i) + ',' + str(human_log) + '\n')
        human_log_f.write(
            "total_human_time: " + str(sum(human_time_list)) + " , total_human_num: " + str(env.human_decision_num)
            + " , confirm_num: " + str(env.human_confirm_num) + " , reject_num: " + str(env.human_reject_num))
        human_log_f.close()

        reward_list.append(env.total_reward)
        loss_list.append(env.total_loss)
        q_online_list.append(env.q_online)
        q_target_list.append(env.q_target)
        completion_rate_list.append((env.complete_tasks_num / env.total_tasks_num) * 100)

        del hidden

        hidden=None

        gc.collect()
        torch.cuda.empty_cache()

        env.reset()

    print('Simulator Off')
    return QLearning.q_table


if __name__=="__main__":
    start_time=time.time()
    worker_num=100
    worker_capacity=20
    task_num=10000
    epoch_num = 5
    data_set='CD'

    actions_num=len(action_space)

    #Building a DARQN network
    online_net = DARQN(grid_num,feature_num, actions_num)
    target_net = DARQN(grid_num,feature_num, actions_num)
    update_target_model(online_net, target_net)

    optimizer = optim.RMSprop(online_net.parameters(), lr=lr)
    # writer = SummaryWriter('logs')

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity)
    loss_list=[]
    reward_list=[]
    q_online_list=[]
    q_target_list=[]

    completion_rate_list = []

    env=Env(worker_num,worker_capacity,task_num,grid_num)
    # Load supervised model
    if os.path.exists('./src/qTable/q_table_DRQN.pkl'):
        table = pd.read_pickle('./src/qTable/q_table_DRQN.pkl')
        print('Loading Q_table...\n', table)
        QLearning = QLearningTable(q_table=table, actions=list([0, 1]))
    else:
        print('Q_table cannot find!')
        table = pd.DataFrame(columns=list([0, 1]), dtype=np.float64)
        QLearning = QLearningTable(q_table=table, actions=list([0, 1]))

    table=Simulation(env,epoch_num,online_net,target_net,optimizer,memory,loss_list,reward_list,q_online_list,q_target_list,QLearning,completion_rate_list)
    table.to_pickle('./src/qTable/q_table_DRQN.pkl')


    fig_dir = "./results/figs_human/"
    data_dir = "./results/data_human/"

    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.ylabel('DARQN Loss')
    plt.xlabel('epochs')
    loc_time_loss = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    save_path = fig_dir+data_set+"_gridNum_"+str(grid_num)+"_lr_"+str(lr)+"_batchSize_"+str(batch_size)+"_DARQN_loss.png"
    # print(save_path)
    foo_fig = plt.gcf()
    foo_fig.savefig(save_path, dpi=600, format='png')
    plt.close()

    plt.plot(np.arange(len(q_online_list)), q_online_list)
    plt.plot(np.arange(len(q_target_list)), q_target_list)
    plt.ylabel('DARQN Q value')
    plt.xlabel('epochs')
    loc_time_loss = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    save_path = fig_dir+data_set+"_gridNum_"+str(grid_num)+"_lr_"+str(lr)+"_batchSize_"+str(batch_size)+"_DARQN_qValue.png"
    # print(save_path)
    foo_fig = plt.gcf()
    foo_fig.savefig(save_path, dpi=600, format='png')
    plt.close()

    plt.plot(np.arange(len(reward_list)), reward_list)
    plt.ylabel('DARQN Reward')
    plt.xlabel('epochs')
    loc_time_reward = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    save_path = fig_dir+data_set+"_gridNum_"+str(grid_num)+"_lr_"+str(lr)+"_batchSize_"+str(batch_size)+"_DARQN_reward.png"
    # print(save_path)
    foo_fig = plt.gcf()
    foo_fig.savefig(save_path, dpi=600, format='png')
    # plt.show()
    plt.close()

    plt.plot(np.arange(len(completion_rate_list)), completion_rate_list)
    plt.ylabel('DARQN completion Rate')
    plt.xlabel('epochs')
    loc_time_reward = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    save_path = fig_dir + data_set + "_gridNum_" + str(grid_num) + "_lr_" + str(lr) +"_batchSize_"+str(batch_size)+ "_DARQN_completionRate.png"
    # print(save_path)
    foo_fig = plt.gcf()
    foo_fig.savefig(save_path, dpi=600, format='png')
    # plt.show()
    plt.close()

    q_target_data_file = data_dir + data_set + "_gridNum_" + str(grid_num) + "_lr_" + str(lr)+"_batchSize_"+str(batch_size)+ "_DARQN_qTarget.txt"
    q_online_data_file = data_dir + data_set + "_gridNum_" + str(grid_num) + "_lr_" + str(lr) +"_batchSize_"+str(batch_size)+ "_DARQN_qOnline.txt"
    loss_data_file = data_dir + data_set + "_gridNum_" + str(grid_num) + "_lr_" + str(lr)+"_batchSize_"+str(batch_size)+ "_DARQN_loss.txt"
    reward_data_file = data_dir + data_set + "_gridNum_" + str(grid_num) + "_lr_" + str(lr) +"_batchSize_"+str(batch_size)+ "_DARQN_reward.txt"
    completion_rate_data_file = data_dir + data_set + "_gridNum_" + str(grid_num) + "_lr_" + str(lr)+"_batchSize_"+str(batch_size) + "_DARQN_completionRate.txt"

    online_f = open(q_online_data_file, "w")
    for i in range(len(q_online_list)):
        q = q_online_list[i]
        online_f.write(str(i) + ',' + str(q) + '\n')
    online_f.close()

    target_f = open(q_target_data_file, "w")
    for i in range(len(q_target_list)):
        q = q_target_list[i]
        target_f.write(str(i) + ',' + str(q) + '\n')
    target_f.close()
    
    loss_f=open(loss_data_file,"w")
    for i in range(len(loss_list)):
        loss=loss_list[i]
        loss_f.write(str(i)+','+str(loss)+'\n')
    loss_f.close()

    reward_f = open(reward_data_file, "w")
    for i in range(len(reward_list)):
        reward = reward_list[i]
        reward_f.write(str(i) + ',' + str(reward) + '\n')
    reward_f.close()

    completion_f = open(completion_rate_data_file, "w")
    for i in range(len(completion_rate_list)):
        completion_rate = completion_rate_list[i]
        completion_f.write(str(i) + ',' + str(completion_rate) + '\n')
    completion_f.close()

    print("grid_num:",grid_num)

    print("current time:%s, total cost time:%s, average training time:%s" %
          (loc_time_loss, time.time()-start_time,((time.time()-start_time)/epoch_num)))

    print('save model successfully! ')
