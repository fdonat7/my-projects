# -*- coding: utf-8 -*-

#!/usr/bin/python3

import random
from queue import Queue, PriorityQueue
import numpy as np
import matplotlib.pyplot as plt
import time as t
import math as m

# ******************************************************************************
# Constants
# ******************************************************************************
# constants used for decide and generate values about load/interarrival times and service time
SERVICE = 15.0  # av service time DEFAULT=10
IS_LOAD_FIXED = 0  # if it is =0, the ARRIVAL is independent from LOAD and LOAD depends on it
FIXED_LOAD = 0.85
FIXED_ARRIVAL = SERVICE/FIXED_LOAD
ARRIVAL_RATE_NUMBER = 12
STATISTICS_FREQUENCY = 12
FIRST_ARRIVAL_RATE = 30 #10 or 45.5 or 50 or 20 #in reality it is the first inter arrvial time
FIRST_HOUR = 13
ARRIVAL_RATE_RATIO = 1.5  # 2.5
LUNCH_ARRIVAL_RATE_RATIO = 1.3  # 2.5
SERVICE_TYPE_DISTRIBUTION = 'exp'  # can be 'exp, 'uniform' or 'tcp'

# constant used for defining different buffer sizes to be tested
# choose -1 to define an infinite buffeer
BUFFER_SIZE = 0, 10 #,50, 1000 ,999999999      #buffer sizes tested
IS_BUFFER_UNIQUE = 0  # if it is =0, each server has its own queue;
# to be set also if each server has not a buffer, but does not exist a uniwue buffer (SO NO BUFFER AT ALL) and the server is chosen randomly i.e. each server is on a different drone
# if it is =1, we have one buffer for all servers;
# if it is =2, we have many buffers for a single server
BSbuffer = 100
BUFFERS_NUMBER = 3
# constant used for defining the number of servers
SERVERS_NUMBER = 3
SERVERS_TYPES=['BS','TYPE0W', 'TYPE0W']  #to define the type of the used servers
ANTENNAS_NUMBER = [2,1,1]             #to define the number of antennas for each server THE FIRST ELEMENT IS THE BS

# constants useful for runs settings
SIM_TIME = 500000  #0
TYPE1 = 1
RUNS_NUMBER = 20
FIRST_SEED = 37
SEED_INTERVAL = 17

SCHED_STRAT = 1    # 1 -> LOSSES   2 -> ARRIVAL RATE  3->buffer  4->losses + arr rate    *any other number*-> no sched strat

REAL_TIME = 1 #if =0 when you launch the drone, it stays for 25 mins even when BS does not need it
# scheduling strategies constants
LOSS_THRESH = 110
ARR_THRESH = 0.05
THRESHOLD_FOR_DRONE_BUFFER = 0.75 #percentage of buffer occupance that when overcome enalbles the sending of another drone
STRAT_ONE_PERCENTAGE = 0.2 #if in interval i we have 20 losses and in i+1 we have STRAT_ONE_PERCENTAGE*20 losses the drone starts

# drones management
MAX_BATTERY_CAPACITY_0W = 5
MAX_BATTERY_CAPACITY_40W = 6
MAX_BATTERY_CAPACITY_60W = 7
MAX_BATTERY_CAPACITY_70W = 8
REST_BATTERY_TIME = 12
MAXIMUM_RECHARGING_CYCLES = 5

# conntants used for deploying the usage of Confidence Interval
CI_ARRIVAL_RATE_index = 16  # CI stands for Confidence Interval
CI_BUFF_SIZE_index = 1


# 90% confidence interval with 25 runs and no buffer, from t-student table: alfa=0.10 and g-1=RUNS_NUMBER-1=24 -> ts=1.711
TS = 1.711
# ******************************************************************************
# To take the measurements
# ******************************************************************************


class Measure:
    def __init__(self, Narr, Ndep, NAveraegUser, OldTimeEvent, AverageDelay, TimeInTheSystem, DelayInQueue, losses, total_users):
        self.arr = Narr
        self.dep = Ndep
        self.ut = NAveraegUser
        self.delay = AverageDelay
        self.time_in_the_system = TimeInTheSystem
        self.delay_in_queue = DelayInQueue
        self.losses = losses
        self.drones_usage = np.zeros(SERVERS_NUMBER)
        self.total_users = total_users
        self.drones_users = np.zeros(SERVERS_NUMBER)
        self.drones_buffer_usage = np.zeros(SERVERS_NUMBER)
        self.drones_users = np.zeros(SERVERS_NUMBER)
        self.drones_avg_buffer_usage = np.zeros(SERVERS_NUMBER)
        self.arrivals=np.zeros(SERVERS_NUMBER)
        self.maximum_contemp_antennas = np.zeros(SERVERS_NUMBER)



class GeneralMeasure:
    def __init__(self, Narr, Ndep, NAveraegUser, OldTimeEvent, AverageDelay, TimeInTheSystem, DelayInQueue, losses):
        self.arr = Narr
        self.dep = Ndep
        self.ut = NAveraegUser
        self.oldT = OldTimeEvent
        self.delay = AverageDelay
        self.time_in_the_system = TimeInTheSystem
        self.delay_in_queue = DelayInQueue
        self.losses = losses
        self.users_in_drone_buffer = []
        for s in range(SERVERS_NUMBER):
            self.users_in_drone_buffer.append([])


# ******************************************************************************
# Client
# ******************************************************************************


class Client:
    def __init__(self, type, arrival_time):
        self.type = type
        self.arrival_time = arrival_time

# ******************************************************************************
# Server
# ******************************************************************************


class Server(object):

    # constructor
    def __init__(self, buffer_size, isActive, drone_type,antennas_number):
        # whether the server is idle or not
        self.antennas_number=antennas_number
        self.idle = self.antennas_number               #it is not a binary value as it is possible to have many antennas
        self.service = random.uniform(-5, 5)+SERVICE
        self.departure = -1
        self.buffer_occupance = 0
        self.buffer_size = buffer_size
        self.queue = []
        self.total_buffer_occupance = 0
        self.total_event = 0
        self.busy_time = 0
        self.delay_in_queue = 0
        self.buffer_queue = 0
        self.isActive = isActive
        #to set the type of drone, defining battery capacity, mmultiplying factor for service rate or buffer size
        #battery capacity
        if drone_type=='BS':
            self.service=self.service*2.5   #default *2
            self.MAX_BATTERY_CAPACITY=MAX_BATTERY_CAPACITY_0W
        elif drone_type=='TYPE0W':
            self.MAX_BATTERY_CAPACITY = MAX_BATTERY_CAPACITY_0W
        elif drone_type=='TYPE40W':
            self.MAX_BATTERY_CAPACITY = MAX_BATTERY_CAPACITY_40W
        elif drone_type=='TYPE60W':
            self.MAX_BATTERY_CAPACITY = MAX_BATTERY_CAPACITY_60W
            self.service=self.service/2
        elif drone_type=='TYPE70W':
            self.MAX_BATTERY_CAPACITY = MAX_BATTERY_CAPACITY_70W
        #battery capacity + dobule service rate
        elif drone_type=='TYPE0W_2S':
            self.MAX_BATTERY_CAPACITY = MAX_BATTERY_CAPACITY_0W
            self.service = 2*self.service
        elif drone_type=='TYPE40W_2S':
            self.MAX_BATTERY_CAPACITY = MAX_BATTERY_CAPACITY_40W
            self.service = 2*self.service
        elif drone_type=='TYPE60W_2S':
            self.MAX_BATTERY_CAPACITY = MAX_BATTERY_CAPACITY_60W
            self.service = 2*self.service
        elif drone_type=='TYPE70W_2S':
            self.MAX_BATTERY_CAPACITY = MAX_BATTERY_CAPACITY_70W
            self.service = 2*self.service
        #battery capacity + double buffer size
        elif drone_type=='TYPE0W_2B':
            self.MAX_BATTERY_CAPACITY = MAX_BATTERY_CAPACITY_0W
            self.buffer_size = 2*self.buffer_size
        elif drone_type=='TYPE40W_2B':
            self.MAX_BATTERY_CAPACITY = MAX_BATTERY_CAPACITY_40W
            self.buffer_size = 2*self.buffer_size
        elif drone_type=='TYPE60W_2B':
            self.MAX_BATTERY_CAPACITY = MAX_BATTERY_CAPACITY_60W
            self.buffer_size = 2*self.buffer_size
        elif drone_type=='TYPE70W_2B':
            self.MAX_BATTERY_CAPACITY = MAX_BATTERY_CAPACITY_70W
            self.buffer_size = 2*self.buffer_size
        self.charge = self.MAX_BATTERY_CAPACITY
        self.rest = 0
        self.rechargingCycles = MAXIMUM_RECHARGING_CYCLES #maximum number of daily cycles for the battery 
        self.total_served_clients = 0

    def getIdle(self):
        return self.idle>0

    def setIdle(self):
        self.idle += 1

    def setBusy(self, event,i,s_index):
        self.idle -= 1
        val=self.antennas_number-self.idle
        if val>data[i].maximum_contemp_antennas[s_index]:
            data[i].maximum_contemp_antennas[s_index]=val
        if event == 'departure':
            self.buffer_occupance -= 1

    def getService(self):
        return self.service

    def setDeparture(self, departure):
        self.departure = departure

    def getDeparture(self):
        return self.departure

    def insertInBuffer(self, client, i, s):
        self.buffer_occupance += 1
        self.buffer_queue += 1
        self.queue.append(client)
        data[i].drones_buffer_usage[s] += 1

    def popFromQueue(self):
        return self.queue.pop(0)

    def hasBufferSpace(self):
        return self.buffer_occupance < self.buffer_size

    def isQueueEmpty(self):
        return len(self.queue) == 0

    def initialize(self, isActive):   #used to reinitialize servers before a new run
        self.deaprture = -1
        self.idle = self.antennas_number
        self.buffer_occupance = 0
        self.queue = []
        self.total_buffer_occupance = 0
        self.total_event = 0
        self.busy_time = 0
        self.delay_in_queue = 0
        self.buffer_queue = 0
        self.isActive = isActive
        self.rest = 0
        self.charge = self.MAX_BATTERY_CAPACITY
        self.rechargingCycles = MAXIMUM_RECHARGING_CYCLES
        self.total_served_clients = 0

    def bufferOccupanceIncrement(self):
        self.total_event += 1
        self.total_buffer_occupance += self.buffer_occupance

    def getBufferOccupanceInfo(self):
        return self.total_buffer_occupance, self.total_event

    def getTotalBusyTime(self):
        return self.busy_time

    def busyTimeIncrement(self, time):
        self.busy_time += time

    def addDelayInQueue(self, time):
        self.delay_in_queue += time

    def getDelayInQueue(self):
        return self.delay_in_queue

    def getBufferUsers(self):
        return self.buffer_queue

    def getActive(self):
        return self.isActive

    def setActive(self):
        self.isActive = 1

    def deActivate(self, i):
        self.isActive = 0
        data[i].losses += len(self.queue)+self.antennas_number-self.idle
        self.buffer_occupance = 0
        self.queue = []
        self.buffer_queue = 0

    def getBufferOccupance(self):
        return self.buffer_occupance

    def reCharge(self):
        if self.charge < self.MAX_BATTERY_CAPACITY:
            self.rest += 1

    def isCharge(self): #to check if the drone is charged, cosidering also the maximum number of rechargin cycles 
        if self.rest == REST_BATTERY_TIME and self.rechargingCycles > 0:
            self.charge = self.MAX_BATTERY_CAPACITY
            self.rest = 0
            self.rechargingCycles -= 1
            return True
        elif self.charge > 0:
            return True
        elif self.rechargingCycles == 0:
            print('Drone no more usable for today')
        return False

    def deCharge(self, i):
        self.charge -= 1
        if self.charge == 0:
            self.deActivate(i)

    def addServedClient(self):
        self.total_served_clients += 1

    def getServedClients(self):
        return self.total_served_clients
    
    def getBufferSize(self):
        return self.buffer_size


# ******************************************************************************

# arrivals *********************************************************************

def tcpDistribution():
    choice = random.randint(0, 100)
    if choice < 49:
        length = 40
    elif choice == 49 or choice == 50:
        length = random.uniform(41, 1459)
    else:
        length = 1460
    return SERVICE/750*random.gauss(length, 1.0)  # GAUSSIAN

# ******************************************************************************

# arrivals *********************************************************************


def arrival(time, FES, queue, servers, buff, a, buffer_queue, th):
    global users  # this is the total number of users in the system, considering all buffers, all drones and ll servers
    global losses  # this is the total number of losses in the system, considering all buffers, all drones and ll servers
    global idle_servers  # used to make a double check in some situations
    # print("Arrival no. ",data.arr+1," at time ",time," with ",users," users" )
    global users_in_queue
    global total_event
    global total_buffer_occupance
    global drone_arrival_in_the_slots
    global old_slot

    slot_updated = 0

    for i in range(len(HOURS)-1):
        if time/SIM_TIME*ARRIVAL_RATE_NUMBER+FIRST_HOUR >= HOURS[i] and time/SIM_TIME*ARRIVAL_RATE_NUMBER+FIRST_HOUR < HOURS[i+1]:
            if i > old_slot:
                old_slot = i
                slot_updated = 1
            break
    # ☻print('slot=',i)
    data[i].arr += 1
    data[i].total_users += 1
    data[i].ut += (time-generalData.oldT)
    generalData.arr += 1
    generalData.ut += (time-generalData.oldT)
    generalData.oldT = time

    for s in range(len(servers)):
        data[i].drones_usage[s] = servers[s].getActive()
        data[i].drones_avg_buffer_usage[s]+=servers[s].getBufferOccupance()
        data[i].arrivals[s]+=1

    # sample the time until the next event
    inter_arrival = random.expovariate(lambd=1.0/a)

    # schedule the next arrival
    # print('at time', time, 'scehduling an arrival at',time+inter_arrival)
    FES.put((time + inter_arrival, "arrival"))
    #print('arrival scheduled at', time+inter_arrival)

    if IS_BUFFER_UNIQUE == 0:

        no_loss = 0
        s_index = -1
        for s in servers:
            s_index += 1
            if s.getActive() == 1 and (s.hasBufferSpace() == True or s.getIdle() == True):
                no_loss = 1
                break

        if no_loss == 0:
            losses += 1
            data[i].losses += 1  # LOSSES IN THE i-th SLOT
            generalData.losses += 1
            # print('LOSS')
            #print(s_index, SERVERS_NUMBER)

        if slot_updated == 1:  # the system does not react rel-time, but once a time slot
            #print('slot updated')

            # updating servers state: recharging or decharging them
            for ser in range(1, SERVERS_NUMBER):
                #print(ser, data[i].drones_usage[ser] )
                if servers[ser].getActive():
                    #print('Active server:', ser, i)
                    servers[ser].deCharge(i)
                    data[i].drones_usage[ser] += 1
                else:
                    servers[ser].reCharge()
            s_index = -1
            for s in servers:
                s_index += 1
                if s.getActive() == False and s.isCharge() == True:
                    break
            if s_index < SERVERS_NUMBER:

                if SCHED_STRAT == 1:
                    #print('\nIM IN THE STRATEGY', 'data[i-2].losses=', data[i-2].losses, 'data[i-1].losses=', data[i-1].losses, 'second condition=',data[i-1].losses>=STRAT_ONE_PERCENTAGE*data[i-2].losses ,'s.isCharge()=', s.isCharge(), 's_index=', s_index )
                    # t.sleep(10)
                    if data[i-2].losses > th and data[i-1].losses >= STRAT_ONE_PERCENTAGE*data[i-2].losses and s.isCharge():
                        s.setActive()
                        #print('\n!!! has been activated:', s)
                    elif data[i-1].losses <= th and REAL_TIME == 1:
                        s.deActivate(i)
                        #print('1n!!! has been deactivated:', s)

                        #print("CIAO", s_index, capacity)

                elif SCHED_STRAT == 2:
                    if 1/a > th and s.isCharge():
                        #print("CIAO", s_index)
                        s.setActive()
                        #print("arrival rate=",1/a)
                    elif 1/a < th and REAL_TIME==1:
                        s.deActivate(i)
                        
                elif SCHED_STRAT == 3:
                    if buff > 0 and servers[s_index-1].getBufferOccupance() > THRESHOLD_FOR_DRONE_BUFFER*servers[s_index-1].getBufferSize() and s.isCharge():
                        s.setActive()
                    elif buff > 0 and servers[s_index-1].getBufferOccupance() < THRESHOLD_FOR_DRONE_BUFFER*servers[s_index-1].getBufferSize() and REAL_TIME == 1:
                        s.deActivate(i)
                        
                elif SCHED_STRAT==4:
                    if data[i-2].losses > th and data[i-1].losses >= STRAT_ONE_PERCENTAGE*data[i-2].losses and 1/a > ARR_THRESH and s.isCharge():
                        s.setActive()
                    elif 1/a < ARR_THRESH and data[i-1].losses <= th and REAL_TIME==1:
                        s.deActivate(i)

        s_index = -1
        for s in servers:
            s_index += 1
            if s.getActive() == 1 and (s.hasBufferSpace() == True or s.getIdle() == True):
                #print("AO", s_index)
                break

        # print(s_index)
        drone_arrival_in_the_slots[s_index, i] += 1
        if no_loss == 1:
            #print('assigning the user to the server', s_index, 'that has buffer occupance=', servers[s_index].getBufferOccupance(), 'over', servers[s_index].buffer_size)
            if s.hasBufferSpace() == True and s.getIdle() == False:
                users += 1
                users_in_queue += 1
                client = Client(TYPE1, time)
                s.insertInBuffer(client, i, s_index)
                #print('putting user in buffer')

            elif s.getIdle() == True and s.isQueueEmpty() == True:
                #print('putting user in service')
                s.setBusy('arrival',i,s_index)
                idle_servers -= 1
                service_t = s.getService()
                users += 1
                # sample the service time basing on the chosen definition
                # we assume to have the same distribution for all servers but different servite times
                if SERVICE_TYPE_DISTRIBUTION == 'exp':
                    service_time = random.expovariate(1.0/service_t)
                elif SERVICE_TYPE_DISTRIBUTION == 'uniform':
                    service_time = 1 + random.uniform(0, service_t)
                elif SERVICE_TYPE_DISTRIBUTION == 'tcp':
                    service_time = tcpDistribution()
                data[i].delay += service_time
                generalData.delay += service_time
                # schedule when the client will finish the server

                s.setDeparture(time+service_time)
                if time+service_time < SIM_TIME:
                    s.busyTimeIncrement(service_time)
                FES.put((time + service_time, "departure"))
                #print('departure scheduled at', time+service_time)


# ******************************************************************************

# departures *******************************************************************
def departure(time, FES, queue, servers, buffer_queue):
    global users  # this is the total number of users in the system, considering all buffers, all drones and ll servers
    global idle_servers  # used to make a double check in some situations
    # print("Departure no. ",data.dep+1," at time ",time," with ",users," users" )
    global users_in_queue
    global arrival_in_the_slots
    global losses_in_the_slots

    # cumulate statistics
    for i in range(len(HOURS)-1):
        if time/SIM_TIME*ARRIVAL_RATE_NUMBER+FIRST_HOUR >= HOURS[i] and time/SIM_TIME*ARRIVAL_RATE_NUMBER+FIRST_HOUR < HOURS[i+1]:
            break

    data[i].dep += 1
    data[i].total_users -= 1
    data[i].ut += users*(time-generalData.oldT)
    generalData.oldT = time
    generalData.dep += 1
    generalData.ut += users*(time-generalData.oldT)

    if IS_BUFFER_UNIQUE == 0:
        users -= 1
        s_index = 0
        for s in servers:
            if s.getDeparture() == time and s.getActive()==0:  #due to the server down the packet has already been lost and we don't set the dep
                data[i].dep-=1
                generalData.dep-=1
                return
                                
                
            # print("getDep: ", s.getDeparture(),"time: ", time)
            if s.getIdle() == False and s.getDeparture() == time:
                s.setIdle()
                s.addServedClient()
                data[i].drones_users[s_index] += 1
                idle_servers += 1
                break
            s_index += 1
        s.bufferOccupanceIncrement()
        if s.isQueueEmpty() == False:
            # get the first element from the queue
            client = s.popFromQueue()
            s.addDelayInQueue(time-client.arrival_time)
            # print(f'adding {time-client.arrival_time} to server {s}')
            # print(f'current delay in queue is {s.getDelayInQueue()}')
            # do whatever we need to do when clients go away
            # users -= 1
            s.setBusy('departure',i,s_index)
            idle_servers -= 1
            service_t = s.getService()
            # sample the service time
            service_time = random.expovariate(1.0/service_t)
            # data.time_in_the_system+=(time+service_time-client.arrival_time)
            if time+service_time < SIM_TIME:
                s.busyTimeIncrement(service_time)
            # schedule when the client will finish the server
            FES.put((time + service_time, "departure"))
            s.setDeparture(time+service_time)
            data[i].delay += time+service_time-client.arrival_time
            generalData.delay += time+service_time-client.arrival_time


# ******************************************************************************

# plots *******************************************************************
def plot_all_graphs(HOURS, avg_arrivals, avg_departures, avg_losses, avg_loss_probability,
                    slot_arrivals, slot_departures, slot_losses, slot_loss_probabilities,drones_usage,
                    users_in_drone_buffer, drones_users, total_served_clients, max_antenna_usage,slot_delays, b):

    if b == -1:
        b = 'infinite'
    nn = np.arange(len(HOURS))
    width=0.9/SERVERS_NUMBER
    servers_indexes = []
    for i in range(SERVERS_NUMBER):
        servers_indexes.append(i)
    
    colors=['cyan','green','orange','purple', 'yellow']
    
    # ARRIVALS
    plt.figure(figsize=(6, 4))
    ticks = nn
    fig, ax = plt.subplots()
    ax.plot(HOURS[0:len(HOURS)-2], slot_arrivals[0:len(slot_arrivals)-1])
    ax.set_xticks([HOURS[0], HOURS[12], HOURS[24], HOURS[36], HOURS[48], HOURS[60],
                  HOURS[72], HOURS[84], HOURS[96], HOURS[108], HOURS[120], HOURS[132]])
    plt.xlabel("hour")
    plt.ylabel('arrivals')
    plt.title(f'Arrivals in 5 minutes slots with buffer of size {b}')
    plt.ylim([0, 1.3*max(slot_arrivals)])
    plt.xlim([FIRST_HOUR-0.3,FIRST_HOUR+ARRIVAL_RATE_NUMBER])
    plt.grid()
    plt.tight_layout()
    plt.axhline(y=avg_arrivals, color='g')
    plt.show()

    # LOSSES
    plt.figure(figsize=(6, 4))
    ticks = nn
    fig, ax = plt.subplots()
    ax.plot(HOURS[0:len(HOURS)-2], slot_losses[0:len(slot_losses)-1])
    ax.set_xticks([HOURS[0], HOURS[12], HOURS[24], HOURS[36], HOURS[48], HOURS[60],
                  HOURS[72], HOURS[84], HOURS[96], HOURS[108], HOURS[120], HOURS[132]])
    plt.xlabel("hour")
    plt.ylabel('losses')
    plt.title(f'Losses in 5 minutes slots with drones buffer of size {b}')
    plt.ylim([0, 1.3*max(slot_losses)])
    plt.xlim([FIRST_HOUR-0.3,FIRST_HOUR+ARRIVAL_RATE_NUMBER])
    plt.grid()
    plt.tight_layout()
    plt.axhline(y=avg_losses, color='g')
    plt.show()

    # LOSS PROBABILITY
    plt.figure(figsize=(6, 4))
    ticks = nn
    fig, ax = plt.subplots()
    ax.plot(HOURS[0:len(HOURS)-2],
            slot_loss_probabilities[0:len(slot_loss_probabilities)-1])
    ax.set_xticks([HOURS[0], HOURS[12], HOURS[24], HOURS[36], HOURS[48], HOURS[60],
                  HOURS[72], HOURS[84], HOURS[96], HOURS[108], HOURS[120], HOURS[132]])
    plt.xlabel("hour")
    plt.ylabel('loss probability')
    plt.title(f'Loss probability in in 5 minutes slots with drones buffer of size {b}')
    plt.ylim([0, 1.3*max(slot_loss_probabilities)])
    plt.xlim([FIRST_HOUR-0.3,FIRST_HOUR+ARRIVAL_RATE_NUMBER])
    plt.grid()
    plt.tight_layout()
    plt.axhline(y=avg_loss_probability, color='g')
    plt.show()

    # DEPARTURES
    plt.figure(figsize=(6, 4))
    ticks = nn
    fig, ax = plt.subplots()
    ax.plot(HOURS[0:len(HOURS)-2], slot_departures[0:len(slot_departures)-1])
    ax.set_xticks([HOURS[0], HOURS[12], HOURS[24], HOURS[36], HOURS[48], HOURS[60],
                  HOURS[72], HOURS[84], HOURS[96], HOURS[108], HOURS[120], HOURS[132]])
    plt.xlabel("hour")
    plt.ylabel('departures')
    plt.title(f'Departures in in 5 minutes slots with drones buffer of size {b}')
    plt.ylim([0, 1.3*max(slot_departures)])
    plt.xlim([FIRST_HOUR-0.3,FIRST_HOUR+ARRIVAL_RATE_NUMBER])
    plt.grid()
    plt.tight_layout()
    plt.axhline(y=avg_departures, color='g')
    plt.show()

    # DEPARTURES
    plt.figure(figsize=(6, 4))
    ticks = nn
    fig, ax = plt.subplots()
    ax.plot(HOURS[0:len(HOURS)-2], slot_delays[0:len(slot_delays)-1])
    ax.set_xticks([HOURS[0], HOURS[12], HOURS[24], HOURS[36], HOURS[48], HOURS[60],
                  HOURS[72], HOURS[84], HOURS[96], HOURS[108], HOURS[120], HOURS[132]])
    plt.xlabel("hour")
    plt.ylabel('avg delay')
    plt.title(f'Average delays in in 5 minutes slots with drones buffer of size {b}')
    plt.ylim([0, 1.3*max(slot_delays)])
    plt.xlim([FIRST_HOUR-0.3,FIRST_HOUR+ARRIVAL_RATE_NUMBER])
    plt.grid()
    plt.tight_layout()
    plt.show()
    # DRONES USAGE
    # plt.figure(figsize=(6, 4))
    # ticks = nn
    # fig, ax = plt.subplots()
    # for s in range(SERVERS_NUMBER):
    #     ax.plot(HOURS[0:len(HOURS)-2], drones_usage[s]
    #             [0:len(drones_usage[s])-1])
    # ax.set_xticks([HOURS[0], HOURS[12], HOURS[24], HOURS[36], HOURS[48], HOURS[60],
    #               HOURS[72], HOURS[84], HOURS[96], HOURS[108], HOURS[120], HOURS[132]])
    # plt.xlabel("hour")
    # plt.ylabel('drones usage')
    # plt.title(f'Drones usage in in 5 minutes slots with drones buffer of size {b}')
    # plt.ylim([-0.2, 1.2])
    # plt.grid()
    # plt.tight_layout()
    # plt.show()
    
    for s in range(SERVERS_NUMBER):
        plt.figure(figsize=(6, 2))
        fig, ax = plt.subplots(figsize=(6,2))
        #for s in range(SERVERS_NUMBER):
            # , label=Labels[i])
        colors_all=[]
        for i in range (143):
            colors_all.append(colors[s])
        plt.bar(HOURS[0:143], drones_usage[s], width=0.1, color=colors_all)
        ax.set_xticks([HOURS[0], HOURS[12], HOURS[24], HOURS[36], HOURS[48], HOURS[60],
                      HOURS[72], HOURS[84], HOURS[96], HOURS[108], HOURS[120], HOURS[132]])
        plt.xlabel(f'server {s}')
        plt.title(f'Time activation slots with drones buffer of size {b}')
        #plt.legend()
        plt.tight_layout()
        plt.yticks([])
        plt.xlim([FIRST_HOUR-0.3,FIRST_HOUR+ARRIVAL_RATE_NUMBER])
        plt.show()


    # DRONES USERS
    plt.figure(figsize=(6, 4))
    ticks = nn
    fig, ax = plt.subplots()
    for s in range(SERVERS_NUMBER):
        ax.plot(HOURS[0:len(HOURS)-2], drones_users[s]
                [0:len(drones_users[s])-1])
    ax.set_xticks([HOURS[0], HOURS[12], HOURS[24], HOURS[36], HOURS[48], HOURS[60],
                  HOURS[72], HOURS[84], HOURS[96], HOURS[108], HOURS[120], HOURS[132]])
    plt.xlabel("hour")
    plt.ylabel('drones processed users (departures)')
    plt.title(f'Drones processed users in in 5 minutes slots with drones buffer of size {b}')
    plt.ylim([-0.5, max(drones_users[0][:]*1.1)])
    plt.xlim([FIRST_HOUR-0.3,FIRST_HOUR+ARRIVAL_RATE_NUMBER])
    plt.grid()
    plt.tight_layout()
    plt.show()

    # DRONES BUFFERS USAGE
    # plt.figure(figsize=(6, 4))
    # ticks = nn
    # fig, ax = plt.subplots()
    # for s in range(SERVERS_NUMBER):
    #     ax.plot(users_in_drone_buffer_time[s], users_in_drone_buffer_values[s])
    # ax.set_xticks([HOURS[0], HOURS[12], HOURS[24], HOURS[36], HOURS[48], HOURS[60],
    #               HOURS[72], HOURS[84], HOURS[96], HOURS[108], HOURS[120], HOURS[132]])
    # plt.xlabel("hour")
    # plt.ylabel('drones buffers usage')
    # plt.title(f'Drones buffers usage in in 5 minutes slots with drones buffer of size {b}')
    # plt.ylim([max(users_in_drone_buffer_values[0][:])*(-0.1),
    #          max(users_in_drone_buffer_values[0][:])*1.1])
    # plt.grid()
    # plt.tight_layout()
    # plt.show()
    
    #
    for s in range(SERVERS_NUMBER):
        plt.figure(figsize=(6, 2))
        fig, ax = plt.subplots(figsize=(6,2))
        #for s in range(SERVERS_NUMBER):
            # , label=Labels[i])
        colors_all=[]
        for i in range (143):
            colors_all.append(colors[s])
        plt.bar(HOURS[0:143], users_in_drone_buffer[s], width=0.1, color=colors_all)
        ax.set_xticks([HOURS[0], HOURS[12], HOURS[24], HOURS[36], HOURS[48], HOURS[60],
                      HOURS[72], HOURS[84], HOURS[96], HOURS[108], HOURS[120], HOURS[132]])
        plt.xlabel(f'server {s}')
        plt.title(f'Drones average buffer usage in in 5 minutes slots with drones buffer of size {b}')
        #plt.legend()
        plt.ylim([0, max(users_in_drone_buffer[s])*1.1])
        plt.xlim([FIRST_HOUR-0.3,FIRST_HOUR+ARRIVAL_RATE_NUMBER])
        plt.tight_layout()
        plt.grid()
        plt.show()
        
        
    #maximum number of antennas used in each slot
    for s in range(SERVERS_NUMBER):
        plt.figure(figsize=(6, 2))
        fig, ax = plt.subplots(figsize=(6,2))
        #for s in range(SERVERS_NUMBER):
            # , label=Labels[i])
        colors_all=[]
        for i in range (143):
            colors_all.append(colors[s])
        plt.bar(HOURS[0:143], max_antenna_usage[s], width=0.1, color=colors_all)
        ax.set_xticks([HOURS[0], HOURS[12], HOURS[24], HOURS[36], HOURS[48], HOURS[60],
                      HOURS[72], HOURS[84], HOURS[96], HOURS[108], HOURS[120], HOURS[132]])
        plt.xlabel(f'server {s}')
        plt.title(f'Drones maximum antennas usage in in 5 minutes slots with drones buffer of size {b}')
        #plt.legend()
        plt.ylim([0, max(max_antenna_usage[s])*1.1])
        plt.xlim([FIRST_HOUR-0.3,FIRST_HOUR+ARRIVAL_RATE_NUMBER])
        plt.tight_layout()
        plt.grid()
        plt.show()

    #SI POTREBBE RAPPRESENTARE QUANDO IL BUFFER E PIENO

    plt.figure()
    for i in range(SERVERS_NUMBER):
        # , label=Labels[i])
        plt.bar(servers_indexes, total_served_clients, width=0.3)
        plt.xticks(np.arange(SERVERS_NUMBER))
    plt.xlabel('server')
    plt.grid()
    plt.yscale('log')
    plt.ylim([-0.1, total_served_clients[0]*1.1])
    plt.title(f'Total served clients with drones buffer of size {b}')
    #plt.legend('server')
    plt.show()



# ******************************************************************************
# the "main" of the simulation
# ******************************************************************************
# definition of interarrival times or load depending on the choise made

if IS_LOAD_FIXED == 0:
    ARRIVALS = np.zeros(ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY)
    HOURS = np.zeros(ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY)
    ARRIVALS[0] = FIRST_ARRIVAL_RATE
    HOURS[0] = FIRST_HOUR
    for h in range(ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY-1):
        HOURS[h+1] = HOURS[h]+1/STATISTICS_FREQUENCY


# DEFINITION OF ARRIVAL RATES IN FUNCTION OF DAYTIME HOUR
    for j in range(1, ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY):
        if HOURS[j] <= 15:
            ARRIVALS[j] = ARRIVALS[0] / \
                (1+((LUNCH_ARRIVAL_RATE_RATIO-1)/STATISTICS_FREQUENCY))

        elif HOURS[j] > 15 and HOURS[j] <= 16:
            ARRIVALS[j] = ARRIVALS[j-1] * \
                (1+((LUNCH_ARRIVAL_RATE_RATIO-1)/STATISTICS_FREQUENCY))

        elif HOURS[j] > 16 and HOURS[j] <= 17:
            ARRIVALS[j] = ARRIVALS[j-1]

        elif HOURS[j] > 17 and HOURS[j] <= 21:
            ARRIVALS[j] = ARRIVALS[j-1] / \
                (1+((LUNCH_ARRIVAL_RATE_RATIO-1)/STATISTICS_FREQUENCY))

        elif HOURS[j] > 21 and HOURS[j] <= 23:
            ARRIVALS[j] = ARRIVALS[j-1]

        elif HOURS[j] > 23:
            ARRIVALS[j] = ARRIVALS[j-1] * \
                (1+((LUNCH_ARRIVAL_RATE_RATIO-1)/STATISTICS_FREQUENCY))

else:
    LOAD = FIXED_LOAD
    ARRIVALS = []
    ARRIVALS.append(FIXED_ARRIVAL)

#print(ARRIVALS)
# creation of different seeds to have more random runs to average on the obtained results
SEEDS = np.zeros(RUNS_NUMBER)
SEEDS[0] = FIRST_SEED
for i in range(RUNS_NUMBER-1):
    SEEDS[i+1] = SEEDS[i]+SEED_INTERVAL

# arrays for confidence intervals
# lower_bound = np.zeros(ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY)
# upper_bound = np.zeros(ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY)
# loss_prob_mean = np.zeros(ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY)

# load_vect = np.zeros(ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY)

# busy_time_plot = np.zeros(
#     (ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY, SERVERS_NUMBER, len(BUFFER_SIZE)))

# #buff_index = 0

# total_arrivals = np.zeros((len(BUFFER_SIZE), len(ARRIVALS)))
# total_losses = np.zeros((len(BUFFER_SIZE), len(ARRIVALS)))
# total_loss_prob = np.zeros((len(BUFFER_SIZE), len(ARRIVALS)))
# departures = np.zeros((len(BUFFER_SIZE), len(ARRIVALS)))
# arrival_rate = np.zeros((len(BUFFER_SIZE), len(ARRIVALS)))
# departure_rate = np.zeros((len(BUFFER_SIZE), len(ARRIVALS)))
# avg_users = np.zeros((len(BUFFER_SIZE), len(ARRIVALS)))
# avg_delay = np.zeros((len(BUFFER_SIZE), len(ARRIVALS)))
# load = np.zeros((len(BUFFER_SIZE), len(ARRIVALS)))

THRESHOLD=[80]#to test a generic threshold
# myFile = open('sample.txt', mode="r+")
b2_min=9999999
b10_min=9999999
b15_min=99999999
for th in THRESHOLD:
    overall_losses=0
    load_vect = np.zeros(ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY)

    busy_time_plot = np.zeros(
        (ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY, SERVERS_NUMBER, len(BUFFER_SIZE)))

    #buff_index = 0

    total_arrivals = np.zeros((len(BUFFER_SIZE), len(ARRIVALS)))
    total_losses = np.zeros((len(BUFFER_SIZE), len(ARRIVALS)))
    total_loss_prob = np.zeros((len(BUFFER_SIZE), len(ARRIVALS)))
    departures = np.zeros((len(BUFFER_SIZE), len(ARRIVALS)))
    arrival_rate = np.zeros((len(BUFFER_SIZE), len(ARRIVALS)))
    departure_rate = np.zeros((len(BUFFER_SIZE), len(ARRIVALS)))
    avg_users = np.zeros((len(BUFFER_SIZE), len(ARRIVALS)))
    avg_delay = np.zeros((len(BUFFER_SIZE), len(ARRIVALS)))
    load = np.zeros((len(BUFFER_SIZE), len(ARRIVALS)))
    buff_index=0
    for b in BUFFER_SIZE:
        random.seed(7)  # to compare among several runs

        servers = []
        servers.append(Server(BSbuffer, 1, SERVERS_TYPES[0], ANTENNAS_NUMBER[0]))
        for s in range(1, SERVERS_NUMBER):
            print(SERVERS_TYPES[s])
            servers.append(Server(b, 0, SERVERS_TYPES[s],ANTENNAS_NUMBER[s])) 

        busytime = np.zeros(SERVERS_NUMBER)
    # if IS_LOAD_FIXED==0:
    #     if SERVERS_NUMBER>1:
    #         LOAD=SERVERS_NUMBER*SERVICE/a
    #         load_vect[arrival_index]=LOAD

    # CAPACITY AND REST INITIALIZATION
        old_slot = 0
        seed = 0
        buffer_queue = []
        total_event = 0
        total_buffer_occupance = 0
        losses = 0
        drone_arrival_in_the_slots = np.zeros(
            (SERVERS_NUMBER, ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY))
        arrivals = 0
        users = 0  # currently in service
        users_in_queue = 0
        idle_servers = SERVERS_NUMBER
        queue = []

        servers[0].initialize(1)
        for s in range(1, SERVERS_NUMBER):
            servers[s].initialize(0)
            # random.seed(SEEDS[seed])

        data = []
        for d in range(ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY-1):
            data.append(Measure(0, 0, 0, 0, 0, 0, 0, 0, 0))
        generalData = GeneralMeasure(0, 0, 0, 0, 0, 0, 0, 0)
            
            # the simulation time
        time = 0

        # the list of events in the form: (time, type)
        FES = PriorityQueue()

        # schedule the first arrival at t=0
        FES.put((0, "arrival"))
        
        old_slot = 0

        # simulate until the simulated time reaches a constant
        while time < SIM_TIME:
        # t.sleep(3)
            for i in range(len(HOURS)-1):
                if time/SIM_TIME*ARRIVAL_RATE_NUMBER+FIRST_HOUR >= HOURS[i] and time/SIM_TIME*ARRIVAL_RATE_NUMBER+FIRST_HOUR < HOURS[i+1]:
                    a = ARRIVALS[i]
                    load[buff_index, i] = SERVICE/a
                # if i>old_slot:
                #     old_slot=i
                #     for s in range(SERVERS_NUMBER):
                #         data[i].
            (time, event_type) = FES.get()
        # myFile.write(('\n    -    at time: '+str(time)+str(event_type)+'\n'))
        # myFile.write('users in the system= '+str(users)+'\n')
        # myFile.write('queue length= '+str(len(queue))+'\n')
        # myFile.write('idle servers= '+str(idle_servers)+'\n')
        # t.sleep(1)
        #print('\n',event_type, time, time/SIM_TIME*ARRIVAL_RATE_NUMBER+FIRST_HOUR)
            if event_type == "arrival":
                arrival(time, FES, queue, servers, b, a, buffer_queue, th)
            elif event_type == "departure":
                    departure(time, FES, queue, servers, buffer_queue)

        if IS_BUFFER_UNIQUE == 0:
            for s in servers:
                total_buffer_occupance_s, total_event_s = s.getBufferOccupanceInfo()
                # myFile.write("Avg buffer occupation: "+ str(total_buffer_occupance_s/total_event_s)+'\n')
                total_buffer_occupance += total_buffer_occupance_s/SERVERS_NUMBER
                total_event += total_event_s/SERVERS_NUMBER

        for s in range(SERVERS_NUMBER):
        # myFile.write('Total busy time: '+ str(servers[s].getTotalBusyTime())+'\n')
            busytime[s] += servers[s].getTotalBusyTime()
        # save obtained results

        avg_arrivals = generalData.arr/(ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY)
        avg_departures = generalData.dep/(ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY)
        avg_losses = generalData.losses/(ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY)
        avg_loss_probability = generalData.losses/generalData.arr
        slot_arrivals = []
        slot_departures = []
        slot_delays = []
        slot_losses = []
        slot_loss_probabilities = []
        drones_usage = np.zeros(
            (SERVERS_NUMBER, ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY-1))
        drones_users = np.zeros(
            (SERVERS_NUMBER, ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY-1))
        
        if b==2:
            if avg_losses<b2_min:
                b2_min=avg_losses
                t2_min=th
        elif b==10:
            if avg_losses<b10_min:
                b10_min=avg_losses
                t10_min=th
        elif b==15:
            if avg_losses<b15_min:
                b15_min=avg_losses
                t15_min=th
                
        total_served_clients = np.zeros(SERVERS_NUMBER)
        for s in range(SERVERS_NUMBER):
            total_served_clients[s] = servers[s].getServedClients()
    
        max_antenna_usage=np.zeros((SERVERS_NUMBER,ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY-1))
        users_in_drone_buffer=np.zeros((SERVERS_NUMBER,ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY-1))
        for i in range(ARRIVAL_RATE_NUMBER*STATISTICS_FREQUENCY-1):
            slot_arrivals.append(data[i].arr)
            slot_losses.append(data[i].losses)
            slot_departures.append(data[i].dep)
            slot_delays.append(data[i].delay/data[i].dep)
            if data[i].arr != 0:
                slot_loss_probabilities.append(data[i].losses/data[i].arr)
            else:
                slot_loss_probabilities.append(0)
            for s in range(SERVERS_NUMBER):
                max_antenna_usage[s][i]=(data[i].maximum_contemp_antennas[s])
                if data[i].arrivals[s] > 0:
                        users_in_drone_buffer[s][i]=(data[i].drones_avg_buffer_usage[s]/data[i].arrivals[s])
                drones_usage[s][i] = data[i].drones_usage[s]
                drones_users[s][i] = data[i].drones_users[s]
        plot_all_graphs(HOURS, avg_arrivals, avg_departures, avg_losses, avg_loss_probability, slot_arrivals, slot_departures, slot_losses,
                                        slot_loss_probabilities, drones_usage, users_in_drone_buffer, drones_users, total_served_clients,max_antenna_usage,slot_delays, b)
        buff_index += 1
    
    #print(sum(ARRIVALS))

        print(f"thresold value= {th}, buffer size={b}, ARRIVALS= {generalData.arr}, AVG LOSSES: {avg_losses}, AVG DEP= {avg_departures}, LOSSES= {generalData.losses}, DEPARTURES= {generalData.dep}, AVG DELAY={generalData.delay/generalData.dep}\n")
        overall_losses+=generalData.losses
    
    
    print(f" total losses={overall_losses}", th) #, b2 min = {b2_min}, t2 min = {t2_min},  b10 min = {b10_min}, t10 min = {t10_min},  b15 min = {b15_min}, t15 min = {t15_min}\n\n")