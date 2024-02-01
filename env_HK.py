import numpy as np
import random
import time
import copy
import math
from geopy.distance import geodesic

class Task(object):
    def __init__(self,index):
        self.index=index
        self.publish_time=None
        self.deadline=None
        self.lng=None
        self.lat=None
        self.payoff=1

class Worker(object):
    def __init__(self,index):
        self.index=index
        self.lng=None
        self.lat=None
        self.capacity=None
        self.schedule=[]
        self.speed=5
        self.remain_distance=0
        self.inset_index=0
        self.min_distance=0

class BufferPool(object):
    def __init__(self):
        self.worker_list = []
        self.task_list = []

class Env(object):
    def __init__(self,worker_num,worker_capacity,task_num,grid_num):

        self.WORKER_NUM=worker_num
        self.WORKER_CAPACITY = worker_capacity
        self.TASK_NUM=task_num
        self.TASK_FILE="./dataset/HK/task_HK_0_24.txt"
        self.ORIGINAL_TIME=1493568000
        self.GRID_NUM=grid_num
        self.MAX_LNG = 110.4116552
        self.MAX_LAT = 20.0790456
        self.MIN_LNG = 110.2516760
        self.MIN_LAT = 19.96475508

        self.time_slice=2*60

        self.max_distance=self.get_distance(self.MIN_LNG,self.MIN_LAT,self.MAX_LNG,self.MAX_LAT)

       
        self.worker_list=[]
    
        self.task_list=[]
        self.all_tasks=[]

     
        for i in range(self.WORKER_NUM):
            worker=Worker("worker_"+str(i))
            worker.capacity=self.WORKER_CAPACITY
            worker.lng=random.uniform(self.MIN_LNG,self.MAX_LNG)
            worker.lat=random.uniform(self.MIN_LAT,self.MAX_LAT)
            self.worker_list.append(worker)

    
        self.all_tasks=self.get_all_tasks(self.TASK_FILE)
      
        self.task_list=random.sample(self.all_tasks,self.TASK_NUM)
        self.task_list=sorted(self.task_list,key=lambda task:task.publish_time)

        self.bufferPool=BufferPool()
        self.total_tasks_num=0
        self.complete_tasks_num=0
        self.total_reward=0
        self.total_loss=0
        self.q_online=0
        self.q_target=0
        self.human_decision_num=0
        self.human_confirm_num = 0
        self.human_reject_num = 0

        self.human_reject_rate=0.99
        self.human_decision_time=0.99

        self.og_worker_list = copy.deepcopy(self.worker_list)
        self.og_task_list = copy.deepcopy(self.task_list)

    def get_all_tasks(self,task_file):
            '''
            Get all tasks from the dataset
            :param order_file: Order file
            :return: A set of tasks
            '''
    
            task_set = []
            f = open(task_file)  
            line = f.readline()  
            line_1 = 0
            while line:
                try:
                    line = line.split(",")
                    line_1 += 1
                    task=Task(line[0])
                    task.publish_time=int(line[1])
                    task.deadline=int(line[2])+60*20
                    task.lng=float(line[3])
                    task.lat=float(line[4])
                    task_set.append(task)
                except KeyError:
                    print("------error line--------:", line_1)
                line = f.readline()
            f.close()
            print("total line:",line_1)
            return task_set

    def get_distance(self,s_lng,s_lat,e_lng,e_lat):
        '''
        Obtain the distance between two points (m)
        :param s_lng:Starting Longitude
        :param s_lat:Starting latitude
        :param e_lng:End Longitude
        :param e_lat:End latitude
        :return:
        '''
        distance = geodesic((s_lat, s_lng), (e_lat, e_lng)).m
        return distance

    def get_dis_time(self,s_lng,s_lat,e_lng,e_lat,worker):
        '''
        Obtain the movement time between two points
        :param s_lng:Starting Longitude
        :param s_lat:Starting latitude
        :param e_lng:End Longitude
        :param e_lat:End latitude
        :param worker: worker
        :return:
        '''
        distance = geodesic((s_lat, s_lng), (e_lat, e_lng)).m
        time=distance/worker.speed
        return time


    def get_env_state_4(self,current_time_step,window_time_slice):
        '''
        Get the current environment status
        :param current_time_step: 
        :param window_time_slice: 
        :return:
        '''
        state=[]

        task_matrix=self.get_task_matrix()
        state+=task_matrix

        worker_matrix=self.get_worker_matrix()
        state+=worker_matrix

        deadline_matrix=self.get_deadline_matrix(current_time_step)
        state+=deadline_matrix

        capacity_matrix=self.get_capacity_matrix()
        state+=capacity_matrix

        state_np = np.array(state)
        state=state_np.reshape(1, -1).tolist()[0]

        day_time_slice = int((current_time_step - self.ORIGINAL_TIME) / self.time_slice)
        state.append(day_time_slice)

        state.append(window_time_slice)


        return state

    def get_env_state(self, current_time_step, window_time_slice):
        '''
        Get the current environment status
        :param current_time_step: 
        :param window_time_slice: 
        :return:
        '''
        state = []

        task_matrix = self.get_task_matrix()
        state.append(task_matrix)

        worker_matrix = self.get_worker_matrix()
        state.append(worker_matrix)

        deadline_matrix = self.get_deadline_matrix(current_time_step)
        state.append(deadline_matrix)

        capacity_matrix = self.get_capacity_matrix()
        state.append(capacity_matrix)


        day_time_slice=int((current_time_step-self.ORIGINAL_TIME)/2)
        day_time_slice_matrix=[[day_time_slice for i in range(self.GRID_NUM)] for j in range(self.GRID_NUM)]
        state.append(day_time_slice_matrix)


        window_time_slice_matrix=[[window_time_slice for i in range(self.GRID_NUM)] for j in range(self.GRID_NUM)]
        state.append(window_time_slice_matrix)

        return state

    def get_grid_x_index(self,lat):
        '''
        Calculate the grid matrix x number (latitude number)
        :param lat: 
        :return:
        '''

        unit=(self.MAX_LAT-self.MIN_LAT)/self.GRID_NUM
        x_index=int((lat-self.MIN_LAT)/unit)
        if x_index>=self.GRID_NUM-1:
            x_index=self.GRID_NUM-1

        return x_index

    def get_grid_y_index(self, lng):
        '''
        Calculate the grid matrix y number (by number)
        :param lat: 
        :return:
        '''
        unit = (self.MAX_LNG - self.MIN_LNG) / self.GRID_NUM
        y_index = int((lng - self.MIN_LNG) / unit)
        if y_index >= self.GRID_NUM - 1:
            y_index = self.GRID_NUM - 1
        return y_index

    def get_worker_matrix(self):
        '''
        Obtain Worker Matrix
        :return:
        '''
    
        worker_matrix = [[0 for i in range(self.GRID_NUM)] for j in range(self.GRID_NUM)]
        worker_list = self.bufferPool.worker_list

        for worker in worker_list:
            
            x_index = self.get_grid_x_index(worker.lat)
            y_index = self.get_grid_y_index(worker.lng)
            worker_matrix[x_index][y_index] += 1

        return worker_matrix

    def get_task_matrix(self):
        '''
        Obtain task matrix (containing the current number of tasks in the grid)
        :return:
        '''
        task_matrix=[[0 for i in range(self.GRID_NUM)] for j in range(self.GRID_NUM)]

        task_list=self.bufferPool.task_list

        for task in task_list:
            x_index=self.get_grid_x_index(task.lat)
            y_index=self.get_grid_y_index(task.lng)
            task_matrix[x_index][y_index]+=1

        return task_matrix


    def get_deadline_matrix(self,current_time_step):
        '''
        Obtain deadline matrix
        :param current_time_step: 
        :return:
        '''
        deadline_matrix = [[0 for i in range(self.GRID_NUM)] for j in range(self.GRID_NUM)]

        task_list = self.bufferPool.task_list

        for task in task_list:
            x_index = self.get_grid_x_index(task.lat)
            y_index = self.get_grid_y_index(task.lng)
            remain_time=task.deadline-current_time_step
            if remain_time<deadline_matrix[x_index][y_index]:
                deadline_matrix[x_index][y_index] =remain_time

        return deadline_matrix

    def get_capacity_matrix(self):
        '''
        Obtain capacity matrix
        :return:
        '''
        capacity_matrix = [[0 for i in range(self.GRID_NUM)] for j in range(self.GRID_NUM)]

        worker_list = self.bufferPool.worker_list

        for worker in worker_list:
            x_index = self.get_grid_x_index(worker.lat)
            y_index = self.get_grid_y_index(worker.lng)
            capacity_matrix[x_index][y_index] += worker.capacity-len(worker.schedule)

        return capacity_matrix

    def get_window_task(self,current_time_step):
        '''
        Get new tasks
        :param current_time_step:
        :return:
        '''
        while True:
            if len(self.task_list)>0:
                task=self.task_list[0]
                if task.publish_time<=current_time_step:
                    self.bufferPool.task_list.append(task)
                    self.task_list.remove(task)
                    self.total_tasks_num+=1
                else:
                    break
            else:
                break

    def update_buffer_pool(self,current_time_step):
        '''
        Update the buffer pool, with a focus on refreshing the courier (left) node and checking if the request task has expired. Once it expires, remove it
        :param current_time_step: 
        :return:
        '''
        self.bufferPool.worker_list=[]
        temp_task_list = self.bufferPool.task_list
        self.bufferPool.task_list=[]
        for w in self.worker_list:
            if len(w.schedule) < w.capacity:
                self.bufferPool.worker_list.append(w)
        for t in temp_task_list:
            if current_time_step<t.deadline:
                self.bufferPool.task_list.append(t)

    def update_worker_location(self):
        '''
        Update all worker positions
        :return:
        '''

        for worker in self.worker_list:
            move_distance=self.time_slice*worker.speed
            if len(worker.schedule)==0 and worker.remain_distance==0:
                pass
            elif len(worker.schedule)==0 and worker.remain_distance!=0:
                if move_distance>=worker.remain_distance:
                    worker.remain_distance=0
                else:
                    worker.remain_distance-=move_distance
            else:
            
                moved_distance = worker.remain_distance
                while move_distance>0:
                    if move_distance>=worker.remain_distance:
                        move_distance-=moved_distance
                        task_distance=self.get_distance(worker.lng,worker.lat,worker.schedule[0].lng,worker.schedule[0].lat)
                        if move_distance>task_distance:
                            worker.remain_distance = 0
                            moved_distance=task_distance
                            worker.lng = worker.schedule[0].lng
                            worker.lat = worker.schedule[0].lat
                            worker.schedule.remove(worker.schedule[0])
                            self.complete_tasks_num += 1
                        else:
                            worker.remain_distance = self.get_distance(worker.lng, worker.lat, worker.schedule[0].lng,worker.schedule[0].lat) - (move_distance - worker.remain_distance)
                            worker.lng = worker.schedule[0].lng
                            worker.lat = worker.schedule[0].lat
                            worker.schedule.remove(worker.schedule[0])
                            self.complete_tasks_num += 1

                        
                        if len(worker.schedule)==0:
                            worker.remain_distance=0
                            break


                    else:
                        worker.remain_distance -= move_distance

    def is_complete(self,task,worker,current_time_step):
        '''
        Determine if all other tasks can be completed after inserting this task
        :param task:
        :param worker:
        :param current_time_step:
        :return:
        '''
        task_list=copy.deepcopy(worker.schedule)
        inset_index=0
        min_dis=float("inf")
        start_lat=worker.lat
        start_lng=worker.lng
        for i in range(len(task_list)):
            current_task=task_list[i]
            end_lat=current_task.lat
            end_lng=current_task.lng
            
            dis_1=self.get_distance(start_lng,start_lat,task.lng,task.lat)
            
            dis_2=self.get_distance(task.lng,task.lat,end_lng,end_lat)
        
            dis_3=self.get_distance(start_lng,start_lat,end_lng,end_lat)
            increased_distance=dis_1+dis_2-dis_3
            if increased_distance<min_dis:
                min_dis=increased_distance
                inset_index=i

            start_lng=end_lng
            start_lat=end_lat

        
        increased_distance=self.get_distance(task_list[-1].lng,task_list[-1].lat,task.lng,task.lat)
        if increased_distance < min_dis:
            min_dis = increased_distance
            inset_index = len(task_list)

        task_list.insert(inset_index,task)

        
        pre_time=current_time_step+worker.remain_distance
        for i in range(len(task_list)):
            current_task=task_list[i]
            if pre_time>current_task.deadline:
                return False
            else:
                if i!=len(task_list)-1:
                    pre_time+=self.get_distance(current_task.lng,current_task.lat,task_list[i+1].lng,task_list[i+1].lat)/worker.speed

        worker.inset_index=inset_index
        worker.min_distance=min_dis
        return True

    def is_meet_constraint(self,worker,task,current_time_step):
        '''
        Determine whether the task and the worker meet the matching constraints. Specifically, ensure that after insertion, each previous task of the worker can be completed before the deadline
        :param worker:
        :param task:
        :param current_time_step:
        :return:
        '''
        task_list=worker.schedule
        if len(task_list)==0:
        
            move_time=self.get_dis_time(worker.lng,worker.lat,task.lng,task.lat,worker)
            
            time_2=worker.remain_distance/worker.speed
            if move_time+time_2<task.deadline-current_time_step:
                return True
            else:
                return False
        else:
            if self.is_complete(task,worker,current_time_step):
                return True
            else:
                return False

    def get_min_distance_worker(self,worker_list):
        '''
        Obtain workers who obtain the minimum detour distance
        :param worker_list:
        :return:
        '''
        min_dis=float('inf')
        best_worker=None
        for worker in worker_list:
            if worker.min_distance<min_dis:
                min_dis=worker.min_distance
                best_worker=worker

        return best_worker


    def normalization(self, x, Max, Min):
        x = (x - Min) / (Max - Min)
        return x

    def inset_task(self,worker,task):
        '''
        Insert the task into the appropriate position on the worker's schedule
        :param worker:
        :param task:
        :return: 
        '''
        worker.schedule.insert(worker.inset_index,task)

        path_cost=self.normalization(worker.min_distance, self.max_distance, 0)

        reward=task.payoff-path_cost

        return math.pow(reward,2)

    def greedy_match(self,current_time_step):
        '''
        Greedy matching
        :param current_time_step:
        :return:
        '''
        worker_list=self.bufferPool.worker_list
        task_list=self.bufferPool.task_list
        total_reward=0
        for task in task_list:
            available_worker_list=[]
            for worker in worker_list:
                if self.is_meet_constraint(worker,task,current_time_step):
                    available_worker_list.append(worker)
            if len(available_worker_list)!=0:
                best_worker=self.get_min_distance_worker(available_worker_list)
                total_reward+=self.inset_task(best_worker,task)
                
                self.bufferPool.task_list.remove(task)
                if len(best_worker.schedule)>=best_worker.capacity:
                    worker_list.remove(best_worker)

        return total_reward


    def work_thread(self,divide_flag,current_time_step,window_time_slice):
        '''
        Worker thread, the core of the entire environment, responsible for docking with the simulator
        :param divide_flag: 
        :param current_time_step: 
        :param window_time_slice: 
        :return:
        '''
        self.get_window_task(current_time_step)

        self.update_worker_location()
        self.update_buffer_pool(current_time_step)

        reward=0
        if divide_flag==1:
            reward=self.greedy_match(current_time_step)

        next_state=self.get_env_state(current_time_step,window_time_slice)

        return reward,next_state

    def reset(self):
        '''
        Reset Environment
        :return:
        '''
        
        self.worker_list = []
        self.task_list = []

        for i in range(self.WORKER_NUM):
            worker = Worker("worker_" + str(i))
            worker.capacity = self.WORKER_CAPACITY
            worker.lng = random.uniform(self.MIN_LNG, self.MAX_LNG)
            worker.lat = random.uniform(self.MIN_LAT, self.MAX_LAT)
            self.worker_list.append(worker)

        self.task_list = random.sample(self.all_tasks, self.TASK_NUM)
        self.task_list = sorted(self.task_list, key=lambda task: task.publish_time)

        self.bufferPool = BufferPool()
        self.total_tasks_num = 0
        self.complete_tasks_num = 0
        self.total_reward = 0
        self.total_loss = 0
        self.q_online = 0
        self.q_target = 0

        self.human_decision_num=0
        self.human_confirm_num = 0
        self.human_reject_num = 0
        # self.human_reject_rate = 0.99
        # self.human_decision_time = 0.99