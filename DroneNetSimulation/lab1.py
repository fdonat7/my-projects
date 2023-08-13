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
#constants used for decide and generate values about load/interarrival times and service time
SERVICE = 10.0                        #av service time
IS_LOAD_FIXED=0                       #if it is =0, the ARRIVAL is independent from LOAD and LOAD depends on it
FIXED_LOAD=0.85
FIXED_ARRIVAL=SERVICE/FIXED_LOAD
ARRIVAL_RATE_NUMBER=17
FIRST_ARRIVAL_RATE=50   #45.5
ARRIVAL_RATE_RATIO=1.3   #2.5
SERVICE_TYPE_DISTRIBUTION='exp'       #can be 'exp, 'uniform' or 'tcp'

#constant used for defining different buffer sizes to be tested  
#choose -1 to define an infinite buffeer
BUFFER_SIZE=0, 50      #buffer sizes tested 
IS_BUFFER_UNIQUE=0            #if it is =0, each server has its own queue; 
#to be set also if each server has not a buffer, but does not exist a uniwue buffer (SO NO BUFFER AT ALL) and the server is chosen randomly i.e. each server is on a different drone                                     
                                      #if it is =1, we have one buffer for all servers;                                      
                                      #if it is =2, we have many buffers for a single server 
BUFFERS_NUMBER=1                  
#constant used for defining the number of servers
SERVERS_NUMBER=1

#constants useful for runs settings
SIM_TIME = 50000
TYPE1=1
RUNS_NUMBER=20              
FIRST_SEED=37
SEED_INTERVAL=17

#conntants used for deploying the usage of Confidence Interval 
CI_ARRIVAL_RATE_index=16               #CI stands for Confidence Interval
CI_BUFF_SIZE_index=1

#90% confidence interval with 25 runs and no buffer, from t-student table: alfa=0.10 and g-1=RUNS_NUMBER-1=24 -> ts=1.711
TS=1.711
# ******************************************************************************
# To take the measurements
# ******************************************************************************
class Measure:
    def __init__(self,Narr,Ndep,NAveraegUser,OldTimeEvent,AverageDelay, TimeInTheSystem, DelayInQueue):
        self.arr = Narr
        self.dep = Ndep
        self.ut = NAveraegUser
        self.oldT = OldTimeEvent
        self.delay = AverageDelay
        self.time_in_the_system = TimeInTheSystem
        self.delay_in_queue =DelayInQueue
        
# ******************************************************************************
# Client
# ******************************************************************************
class Client:
    def __init__(self,type,arrival_time):
        self.type = type
        self.arrival_time = arrival_time

# ******************************************************************************
# Server
# ******************************************************************************
class Server(object):

    # constructor
    def __init__(self, buffer_size):
        # whether the server is idle or not
        self.idle = True
        self.service=random.uniform(-5,5)+SERVICE
        self.departure=-1
        self.buffer_occupance=0
        self.buffer_size=buffer_size
        self.queue=[]
        self.total_buffer_occupance=0
        self.total_event=0
        self.busy_time=0
        self.delay_in_queue=0
        self.buffer_queue=0
        
    def getIdle(self):
        return self.idle

    def setIdle(self):
        self.idle=True
        
    def setBusy(self, event):
         self.idle=False
         if event=='departure':
             self.buffer_occupance-=1         
         
    def getService(self):
        return self.service
    
    def setDeparture(self, departure):
        self.departure=departure
        
    def getDeparture(self):
        return self.departure
    
    def insertInBuffer(self, client):
        self.buffer_occupance+=1
        self.buffer_queue+=1
        self.queue.append(client)
        
    def popFromQueue(self):
        return self.queue.pop(0)
        
    def hasBufferSpace(self):
        return self.buffer_occupance<self.buffer_size
    
    def isQueueEmpty(self):
        return len(self.queue)==0
    
    def initialize(self):
        self.deaprture=-1
        self.idle=True
        self.buffer_occupance=0
        self.queue=[]
        self.total_buffer_occupance=0
        self.total_event=0
        self.busy_time=0
        self.delay_in_queue=0
        self.buffer_queue=0
        
    def bufferOccupanceIncrement(self):
        self.total_event+=1
        self.total_buffer_occupance+=self.buffer_occupance
    
    def getBufferOccupanceInfo(self):
        return self.total_buffer_occupance, self.total_event
    
    def getTotalBusyTime(self):
        return self.busy_time
    
    def busyTimeIncrement(self, time):
        self.busy_time+=time
    
    def addDelayInQueue(self,time):
        self.delay_in_queue+=time
        
    def getDelayInQueue(self):
        return self.delay_in_queue
    
    
    def getBufferUsers(self):
        return self.buffer_queue
    
    
    
    
# ******************************************************************************

# arrivals *********************************************************************  
    
def tcpDistribution():
    choice=random.randint(0, 100)
    if choice<49:
        length=40
    elif choice==49 or choice==50:
        length=random.uniform(41, 1459)
    else:
        length=1460
    return SERVICE/750*random.gauss(length, 1.0)##GAUSSIAN
    
# ******************************************************************************

# arrivals *********************************************************************
def arrival(time, FES, queue, servers, buff, a, buffer_queue):
    global users    #this is the total number of users in the system, considering all buffers, all drones and ll servers
    global losses    #this is the total number of losses in the system, considering all buffers, all drones and ll servers
    global idle_servers    #used to make a double check in some situations
    #print("Arrival no. ",data.arr+1," at time ",time," with ",users," users" )
    global users_in_queue
    global total_event
    global total_buffer_occupance
    # cumulate statistics
    data.arr += 1
    data.ut += users*(time-data.oldT)
    data.oldT = time

    # sample the time until the next event
    inter_arrival = random.expovariate(lambd=1.0/a)
    
    # schedule the next arrival
    #print('at time', time, 'scehduling an arrival at',time+inter_arrival)
    FES.put((time + inter_arrival, "arrival"))

    if IS_BUFFER_UNIQUE==0:
        s=servers[random.randint(0, SERVERS_NUMBER-1)] 
        s.bufferOccupanceIncrement()
        
        
        
        if s.hasBufferSpace()==False and s.getIdle()==False:
            losses+=1
            return
        
        elif s.hasBufferSpace()==True and s.getIdle()==False: 
            users+=1
            users_in_queue+=1
            client = Client(TYPE1,time)
            s.insertInBuffer(client)
                            
        elif s.getIdle()==True and s.isQueueEmpty()==True:
            s.setBusy('arrival')
            idle_servers-=1
            service_t=s.getService()
            users+=1
            # sample the service time basing on the chosen definition
            #we assume to have the same distribution for all servers but different servite times
            if SERVICE_TYPE_DISTRIBUTION=='exp':
                service_time = random.expovariate(1.0/service_t)
            elif SERVICE_TYPE_DISTRIBUTION=='uniform':
                service_time = 1 + random.uniform(0, service_t)
            elif SERVICE_TYPE_DISTRIBUTION=='tcp':
                service_time=tcpDistribution()
            data.delay+=service_time
            # schedule when the client will finish the server
            
            s.setDeparture(time+service_time)
            if time+service_time<SIM_TIME:
                s.busyTimeIncrement(service_time)
            FES.put((time + service_time, "departure"))
        
    # check if the buffer has space, otherwise losses
    elif IS_BUFFER_UNIQUE==1: 
        if buff>=0 and len(queue)==buff+SERVERS_NUMBER:
            losses+=1
            #myFile.write('packet lost\n')
            return
        
        # create a record for the client
        client = Client(TYPE1,time)
        # insert the record in the queue
        queue.append(client)
        
        users += 1

        #print('idle_Servers: ',idle_servers)
        # if one server is idle start the service
        if idle_servers>0: 
            ser_index=0
            for s in servers:
                #print('server')
                if s.getIdle()==True:
                    s.setBusy('arrival')
                    idle_servers-=1
                    service_t=s.getService()
                    break
                ser_index+=1
            
            # sample the service time basing on the chosen definition
            #we assume to have the same distribution for all servers but different servite times
            if SERVICE_TYPE_DISTRIBUTION=='exp':
                service_time = random.expovariate(1.0/service_t)
            elif SERVICE_TYPE_DISTRIBUTION=='uniform':
                service_time = 1 + random.uniform(0, service_t)
            elif SERVICE_TYPE_DISTRIBUTION=='tcp':
                service_time=tcpDistribution()
            data.time_in_the_system+=service_time
            # schedule when the client will finish the server
            #myFile.write('ARRIVAL - at time '+ str(time)+ 'scheduling departure at '+ str(time+service_time)+ 'in server '+ str(ser_index)+'\n')
            s.setDeparture(time+service_time)
            if time+service_time<SIM_TIME:
                s.busyTimeIncrement(service_time)
            FES.put((time + service_time, "departure"))
        else:
            users_in_queue+=1
            buffer_queue.append(client)
        #myFile.write('ARRIVAL time= ' + str(time)+'users in queue= '+ str(users_in_queue)+'\n')

    elif IS_BUFFER_UNIQUE==2: 
        chosen_queue=random.randint(0, BUFFERS_NUMBER-1)
        if buff>0 and len(queue[chosen_queue])==buff+1:
            losses+=1
            return
        
        # create a record for the client
        client = Client(TYPE1,time)
        # insert the record in the queue
        queue[chosen_queue].append(client)
        
        users += 1

        #print('idle_Servers: ',idle_servers)
        # if one server is idle start the service
        if idle_servers>0: 
            for s in servers:
                #print('server')
                if s.getIdle()==True:
                    s.setBusy('departure')
                    idle_servers-=1
                    service_t=s.getService()
                    break
            
            # sample the service time basing on the chosen definition
            #we assume to have the same distribution for all servers but different servite times
            if SERVICE_TYPE_DISTRIBUTION=='exp':
                service_time = random.expovariate(1.0/service_t)
            elif SERVICE_TYPE_DISTRIBUTION=='uniform':
                service_time = 1 + random.uniform(0, service_t)
            elif SERVICE_TYPE_DISTRIBUTION=='tcp':
                    service_time=tcpDistribution()
            data.time_in_the_system+=service_time
            # schedule when the client will finish the server
            #print('at time', time, 'scheduling departure at', time+service_time)
            s.setDeparture(time+service_time)
            FES.put((time + service_time, "departure"))
        #else:
        #    users_in_queue+=1
            
# ******************************************************************************

# departures *******************************************************************
def departure(time, FES, queue, servers,buffer_queue):
    global users    #this is the total number of users in the system, considering all buffers, all drones and ll servers
    global idle_servers    #used to make a double check in some situations
    #print("Departure no. ",data.dep+1," at time ",time," with ",users," users" )
    global users_in_queue
    
    # cumulate statistics
    data.dep += 1
    data.ut += users*(time-data.oldT)
    data.oldT = time
    
    if IS_BUFFER_UNIQUE==0:
        users-=1
        for s in servers:
            #print("getDep: ", s.getDeparture(),"time: ", time)
            if s.getIdle()==False and s.getDeparture()==time:                
                s.setIdle()
                idle_servers+=1
                break
        s.bufferOccupanceIncrement()           
        if s.isQueueEmpty() == False :
            # get the first element from the queue
            client = s.popFromQueue()
            s.addDelayInQueue(time-client.arrival_time)
            #print(f'adding {time-client.arrival_time} to server {s}')
            #print(f'current delay in queue is {s.getDelayInQueue()}')
            # do whatever we need to do when clients go away    
            #users -= 1
            s.setBusy('departure')
            idle_servers-=1
            service_t=s.getService()
            # sample the service time
            service_time = random.expovariate(1.0/service_t)
            #data.time_in_the_system+=(time+service_time-client.arrival_time)
            if time+service_time<SIM_TIME:
                s.busyTimeIncrement(service_time)
            # schedule when the client will finish the server
            FES.put((time + service_time, "departure"))
            s.setDeparture(time+service_time)
            data.delay+=time+service_time-client.arrival_time
            
    
    elif IS_BUFFER_UNIQUE==1: 
        if len(queue)>0:
            ser_index=0
            for s in servers:
                #print("getDep: ", s.getDeparture(),"time: ", time)
                if s.getIdle()==False and s.getDeparture()==time:
                    s.setIdle()
                    idle_servers+=1
                    break
                ser_index+=1
                
            # get the first element from the queue
            client = queue.pop(0) 
            data.delay += (time-client.arrival_time)
            #myFile.write('popping from server ' +str(ser_index)+ 'at time '+ str(time)+'\n')
            # do whatever we need to do when clients go away    
            #print(data.delay, data.delay_in_queue, time, client.arrival_time, (time-client.arrival_time))
            users -= 1
    
            # see whether there are more clients to in the line
            if len(buffer_queue)>0 : 
                client=buffer_queue.pop(0)
                data.delay_in_queue += (time-client.arrival_time)
                users_in_queue+=1
                ser_index=0
                for s in servers:
                   
                    if s.getIdle()==True:
                        s.setBusy('departure')
                        idle_servers-=1
                        service_t=s.getService()
                        # sample the service time
                        break
                        ser_index+=1
                service_time = random.expovariate(1.0/service_t)   
                data.time_in_the_system+=(time+service_time-client.arrival_time)
                if time+service_time<SIM_TIME:
                    s.busyTimeIncrement(service_time)
                # schedule when the client will finish the server
                FES.put((time + service_time, "departure"))
                s.setDeparture(time+service_time)
                #myFile.write('DEPARUTRE - at time '+ str(time)+ 'scheduling a departure for '+ str(time+service_time)+ 'in server '+ str(ser_index)+'\n')
        #myFile.write('time=' + str(time)+ 'users in queue= '+ str(users_in_queue)+ 'delay in queue= '+str(data.delay_in_queue)+'\n')
                
    elif IS_BUFFER_UNIQUE==2: 
         servers[0].setIdle()
         for q in queue:
             if len(q)>0:    
                 #print("getDep: ", s.getDeparture(),"time: ", time)
                 # get the first element from the queue
                 client = q.pop(0)
     
                 # do whatever we need to do when clients go away    
                 data.delay += (time-client.arrival_time)
                 users -= 1
     
             # see whether there are more clients to in the line
             #if users > 0 :                        
                 servers[0].setBusy('departure')
                 idle_servers-=1
                 service_t=servers[0].getService()
                                      # sample the service time
                 service_time = random.expovariate(1.0/service_t)
                 data.time_in_the_system+=(time+service_time-client.arrival_time)
                 # schedule when the client will finish the server
                 FES.put((time + service_time, "departure"))
                 servers[0].setDeparture(time+service_time)
                 #print('at time', time, 'scheduling a departure for', time+service_time)
                 break

        
def plot_all_graphs(ARRIVALS, buff_index, plot_total_arrivals, plot_total_losses, plot_departures,
                    plot_arrival_rate, plot_departure_rate, plot_avg_users, plot_avg_delay, plot_avg_delay_in_queue, plot_total_loss_prob, plot_load, plot_avg_buffer_occupation, b,load, busy_time_plot):
    
    if b==-1:
        b='infinite'   
    nn=np.arange(len(ARRIVALS))
    
    #
    plt.figure(figsize=(6,4))
    plt.plot(nn,plot_total_arrivals[:,buff_index],'-o')
    ticks=nn
    plt.xticks(nn,  np.round(1/ARRIVALS,3), rotation=90)
    plt.xlabel("arrival rate")
    plt.ylabel('arrivals')
    plt.title(f'Average arrivals in {RUNS_NUMBER} different runs with buffer of size {b}')
    plt.ylim([0, 1.3*max(plot_total_arrivals[:,buff_index])])
    plt.grid()
    plt.tight_layout()
    plt.show()
        
    #
    plt.figure(figsize=(6,4))
    plt.plot(nn,plot_total_losses[:,buff_index],'-o')
    ticks=nn
    plt.xticks(nn, np.round(1/ARRIVALS,3), rotation=90)
    plt.xlabel("arrival rate")
    plt.ylabel('losses')
    plt.title(f'Average losses in {RUNS_NUMBER} different runs with buffer of size {b}')
    plt.ylim([0, 1.3*max(plot_total_losses[:,buff_index])])
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    #
    plt.figure(figsize=(6,4))
    plt.plot(nn,plot_total_loss_prob[:,buff_index],'-o')
    ticks=nn
    plt.xticks(nn, np.round(1/ARRIVALS,3), rotation=90)
    plt.xlabel("arrival rate")
    plt.ylabel('loss probability')
    plt.title(f'Average loss probability in {RUNS_NUMBER} different runs with buffer of size {b}')
    plt.ylim([0, 1.3*max(plot_total_loss_prob[:,buff_index])])
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    #
    plt.figure(figsize=(6,4))
    plt.plot(nn,plot_departures[:,buff_index],'-o')
    ticks=nn
    plt.xticks(nn,  np.round(1/ARRIVALS,3), rotation=90)
    plt.xlabel("arrival rate")
    plt.ylabel('departures')
    plt.title(f'Average departures in {RUNS_NUMBER} different runs with buffer of size {b}')
    plt.ylim([0, 1.3*max(plot_departures[:,buff_index])])
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    #
    plt.figure(figsize=(6,4))
    plt.plot(nn,plot_arrival_rate[:,buff_index],'-o')
    ticks=nn
    plt.xticks(nn,  np.round(1/ARRIVALS,3), rotation=90)
    plt.xlabel("arrival rate")
    plt.ylabel('arrival rate')
    plt.title(f'Average arrival rate in {RUNS_NUMBER} different runs with buffer of size {b}')
    plt.ylim([0, 1.3*max(plot_arrival_rate[:,buff_index])])
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    #
    plt.figure(figsize=(6,4))
    plt.plot(nn,plot_departure_rate[:,buff_index],'-o')
    ticks=nn
    plt.xticks(nn,  np.round(1/ARRIVALS,3), rotation=90)
    plt.xlabel("arrival rate")
    plt.ylabel('departure rate')
    plt.title(f'Departure rate in {RUNS_NUMBER} different runs with buffer of size {b}')
    plt.ylim([0, 1.3*max(plot_departure_rate[:,buff_index])])
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    #
    plt.figure(figsize=(6,4))
    plt.plot(nn,plot_avg_users[:,buff_index],'-o')
    ticks=nn
    plt.xticks(nn, np.round(1/ARRIVALS,3), rotation=90)
    plt.xlabel("arrival rate")
    plt.ylabel('avg number of users')
    plt.title(f'Average users in {RUNS_NUMBER} different runs with buffer of size {b}')
    plt.ylim([0, 1.3*max(plot_avg_users[:,buff_index])])
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    #
    plt.figure(figsize=(6,4))
    plt.plot(nn,plot_avg_delay[:,buff_index],'-o')
    ticks=nn
    plt.xticks(nn, np.round(1/ARRIVALS,3), rotation=90)
    plt.xlabel("arrival rate")
    plt.ylabel('delay')
    plt.title(f'Average delay in {RUNS_NUMBER} different runs with buffer of size {b}')
    plt.ylim([0, 1.3*max(plot_avg_delay[:,buff_index])])
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    #
    if b!=0:
        plt.figure(figsize=(6,4))
        for i in range(BUFFERS_NUMBER):
            plt.plot(nn,plot_avg_delay_in_queue[:,buff_index,i],'-o')
            ticks=nn
            plt.xticks(nn, np.round(1/ARRIVALS,3), rotation=90)
            plt.xlabel("arrival rate")
            plt.ylabel('delay')
            plt.title(f'Average delay in queue in {RUNS_NUMBER} in buffer {i} different runs with buffer of size {b}')
            plt.ylim([0, 1.3*max(plot_avg_delay_in_queue[:,buff_index,i])]) 
            plt.grid()
            plt.tight_layout()
        plt.show()
        
    #
    plt.figure(figsize=(6,4))
    plt.plot(nn,plot_load[:,buff_index],'-o')
    ticks=nn
    plt.xticks(nn, np.round(1/ARRIVALS,3), rotation=90)
    plt.xlabel("arrival rate")
    plt.ylabel('load')
    plt.title(f'Load in {RUNS_NUMBER} different runs with buffer of size {b}')
    plt.ylim([0, 1.3*max(plot_load[:,buff_index])])
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    #
    if b!=0:
        plt.figure(figsize=(6,4))
        for i in range(BUFFERS_NUMBER):
                
            plt.plot(nn,plot_avg_buffer_occupation[i,:,buff_index],'-o')
            ticks=nn
            plt.xticks(nn, np.round(1/ARRIVALS,3), rotation=90)
            plt.xlabel("arrival rate")
            plt.ylabel('avg buffer occupation')
            plt.title(f'Average buffer occupation in {RUNS_NUMBER} different runs with buffer of size {b}') #CON IS_BUFFER_UNIQUE=1 ESCE MALE IL GRAFICO
            plt.ylim([0, 1.3*max(plot_avg_buffer_occupation[i,:,buff_index])])
            plt.grid()
            plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(6,4))
    for s in range(SERVERS_NUMBER):
        #
        plt.plot(nn,busy_time_plot[:,s, buff_index],'-o')
        ticks=nn
        plt.xticks(nn, np.round(1/ARRIVALS,3), rotation=90)
        plt.xlabel("arrival rate")
        plt.ylabel('avg busy time')
        plt.title(f'Average busy time of servers in {RUNS_NUMBER} different runs with buffer of size {b}')
        plt.ylim([0, 1.3*max(busy_time_plot[:,s, buff_index])])
        plt.grid()
        plt.tight_layout()
    plt.show()
    
    
    
def plot_confidence_interval(arrival_rates, mean, upper_bound, lower_bound, z=1, color='#2187bb', horizontal_line_width=0.0000001):
    
    left =(arrival_rates) - horizontal_line_width / 2
    top = upper_bound
    right = (arrival_rates) + horizontal_line_width / 2
    bottom = lower_bound
    plt.xlim(1.1*(1/ARRIVALS[len(ARRIVALS)-1]))
    plt.plot([str(arrival_rates), str(arrival_rates)], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(str(arrival_rates), mean, 'o', color='#f44336')

# ******************************************************************************
# the "main" of the simulation
# ******************************************************************************

#definition of interarrival times or load depending on the choise made  
if IS_LOAD_FIXED==0:
    ARRIVALS=np.zeros(ARRIVAL_RATE_NUMBER)
    ARRIVALS[0]=FIRST_ARRIVAL_RATE
    for i in range(ARRIVAL_RATE_NUMBER-1):
        ARRIVALS[i+1]=ARRIVALS[i]/ARRIVAL_RATE_RATIO
else:
    LOAD=FIXED_LOAD
    ARRIVALS=[]
    ARRIVALS.append(FIXED_ARRIVAL)
    
#creation of different seeds to have more random runs to average on the obtained results
SEEDS=np.zeros(RUNS_NUMBER)
SEEDS[0]=FIRST_SEED
for i in range(RUNS_NUMBER-1):
    SEEDS[i+1]=SEEDS[i]+SEED_INTERVAL
    
#arrays for confidence intervals
lower_bound=np.zeros(ARRIVAL_RATE_NUMBER)
upper_bound=np.zeros(ARRIVAL_RATE_NUMBER)
loss_prob_mean=np.zeros(ARRIVAL_RATE_NUMBER)

load_vect=np.zeros(ARRIVAL_RATE_NUMBER)
        
busy_time_plot=np.zeros((ARRIVAL_RATE_NUMBER, SERVERS_NUMBER, len(BUFFER_SIZE)))

buff_index=0

total_arrivals=np.zeros((RUNS_NUMBER,len(BUFFER_SIZE), len (ARRIVALS)))
total_losses=np.zeros((RUNS_NUMBER,len(BUFFER_SIZE), len (ARRIVALS)))
total_loss_prob=np.zeros((RUNS_NUMBER,len(BUFFER_SIZE), len (ARRIVALS)))
departures=np.zeros((RUNS_NUMBER,len(BUFFER_SIZE), len (ARRIVALS)))
arrival_rate=np.zeros((RUNS_NUMBER,len(BUFFER_SIZE), len (ARRIVALS)))
departure_rate=np.zeros((RUNS_NUMBER,len(BUFFER_SIZE), len (ARRIVALS)))
avg_users=np.zeros((RUNS_NUMBER,len(BUFFER_SIZE), len (ARRIVALS)))
avg_delay=np.zeros((RUNS_NUMBER,len(BUFFER_SIZE), len (ARRIVALS)))
load=np.zeros((RUNS_NUMBER,len(BUFFER_SIZE), len (ARRIVALS)))


plot_avg_buffer_occupation=np.zeros((BUFFERS_NUMBER,len(ARRIVALS),len(BUFFER_SIZE)))
plot_total_arrivals=np.zeros((len(ARRIVALS),len(BUFFER_SIZE)))
plot_total_losses=np.zeros((len(ARRIVALS),len(BUFFER_SIZE)))
plot_total_loss_prob=np.zeros((len(ARRIVALS),len(BUFFER_SIZE)))
plot_departures=np.zeros((len(ARRIVALS),len(BUFFER_SIZE)))
plot_arrival_rate=np.zeros((len(ARRIVALS),len(BUFFER_SIZE)))
plot_departure_rate=np.zeros((len(ARRIVALS),len(BUFFER_SIZE)))
plot_avg_users=np.zeros((len(ARRIVALS),len(BUFFER_SIZE)))
plot_avg_delay=np.zeros((len(ARRIVALS),len(BUFFER_SIZE)))
plot_avg_delay_in_queue=np.zeros((len(ARRIVALS),len(BUFFER_SIZE), BUFFERS_NUMBER))
plot_load=np.zeros((len(ARRIVALS),len(BUFFER_SIZE)))

myFile = open('sample.txt', mode="r+")   
 
for b in BUFFER_SIZE:
    
    servers=[]
    for i in range(SERVERS_NUMBER):
        servers.append(Server(b))
    
    arrival_index=0
    
    for a in ARRIVALS:
        #t.sleep(20)
        busytime=np.zeros(SERVERS_NUMBER)
        if IS_LOAD_FIXED==1:
            if SERVERS_NUMBER>1:
                LOAD=SERVERS_NUMBER*SERVICE/a
                load_vect[arrival_index]=LOAD
                
            else:
                LOAD = SERVICE/a
                load_vect[arrival_index]=LOAD
                
        else:
            LOAD=SERVICE/a
        seed=0
        myFile.write('arrival'+str(a)+'buffer'+str(b)+'\n')            
        for runs in range(RUNS_NUMBER):

            #myFile.write('run'+str(runs)+'arrival'+str(a)+'buffer'+str(b)+'\n')            
            buffer_queue=[]
            total_event=0
            total_buffer_occupance=0
            losses=0
            arrivals=0
            users=0 #currently in service
            users_in_queue=0
            idle_servers=SERVERS_NUMBER
            queue=[]
            if IS_BUFFER_UNIQUE==2:
                for i in range(BUFFERS_NUMBER):
                    queue.append([])
            for s in servers:
                s.initialize()
            random.seed(SEEDS[seed])
            data = Measure(0,0,0,0,0,0,0)

            # the simulation time 
            time = 0

            # the list of events in the form: (time, type)
            FES = PriorityQueue()
    
            # schedule the first arrival at t=0
            FES.put((0, "arrival"))

            # simulate until the simulated time reaches a constant
            while time < SIM_TIME:
                #t.sleep(3)   
                
                if IS_BUFFER_UNIQUE==1:
                    total_event+=1
                if IS_BUFFER_UNIQUE==1 and len(queue)>SERVERS_NUMBER-idle_servers:                 
                    total_buffer_occupance+=len(queue)-SERVERS_NUMBER+idle_servers
                if IS_BUFFER_UNIQUE==2:
                    total_event+=1
                    buffers_occupation=0
                    for q in queue:
                        buffers_occupation+=len(q)
                    total_buffer_occupance+=buffers_occupation-1
                (time, event_type) = FES.get()
                #myFile.write(('\n    -    at time: '+str(time)+str(event_type)+'\n'))
                #myFile.write('users in the system= '+str(users)+'\n')
                #myFile.write('queue length= '+str(len(queue))+'\n')
                #myFile.write('idle servers= '+str(idle_servers)+'\n')
                if event_type == "arrival":
                    arrival(time, FES, queue, servers, b, a, buffer_queue)
                elif event_type == "departure":
                    departure(time, FES, queue, servers, buffer_queue)
                
            # print output data
            #myFile.write("MEASUREMENTS \n\nNo. of users in the queue: "+str(users)+"\nNo. of arrivals = "+
                  #str(data.arr)+"- No. of departures = "+str(data.dep)+'\n')
            #myFile.write("Load: "+str(SERVICE/a)+'\n')
            #myFile.write("\nArrival rate: "+str(data.arr/time)+" - Departure rate: "+str(data.dep/time)+'\n')
            #myFile.write("\nAverage number of users: "+str(data.ut/time)+'\n')
            #if b>0 and users_in_queue>0:
                #myFile.write("Average waiting delay (effective): "+str(data.delay/users_in_queue)+'\n')
            #myFile.write("Average waiting delay (general): "+str(data.time_in_the_system/data.dep)+'\n')
            if IS_BUFFER_UNIQUE==1:
                #myFile.write("Actual queue size: "+str(len(queue))+'\n')
                if len(queue)>0:
                    ciao=0#myFile.write("Arrival time of the last element in the queue: "+str(queue[len(queue)-1].arrival_time)+'\n')
            else:
                for q in queue:
                    ciao=0#myFile.write("Actual queue size: "+str(len(q))+'\n')
                for q in queue:
                    if len(q)>0:
                        ciao=0#myFile.write("Arrival time of the last element in the queue: "+str(q[len(q)-1].arrival_time)+'\n')
            #myFile.write("Total losses:" +str(losses)+'\n')
            if IS_BUFFER_UNIQUE==0:
                for s in servers:
                    total_buffer_occupance_s, total_event_s = s.getBufferOccupanceInfo()
                    #myFile.write("Avg buffer occupation: "+ str(total_buffer_occupance_s/total_event_s)+'\n')
                    total_buffer_occupance+=total_buffer_occupance_s/SERVERS_NUMBER
                    total_event+=total_event_s/SERVERS_NUMBER
            else:
                ciao=0#myFile.write("Avg buffer occupation: "+ str(total_buffer_occupance/total_event)+'\n')          
            for s in range(SERVERS_NUMBER):
                #myFile.write('Total busy time: '+ str(servers[s].getTotalBusyTime())+'\n')
                busytime[s]+=servers[s].getTotalBusyTime()
            #save obtained results
            total_arrivals[runs,buff_index,arrival_index]=data.arr
            total_losses[runs,buff_index,arrival_index]=losses
            total_loss_prob[runs, buff_index, arrival_index]=losses/data.arr
            departures[runs,buff_index,arrival_index]=data.dep
            arrival_rate[runs,buff_index,arrival_index]=data.arr/time
            departure_rate[runs,buff_index,arrival_index]=data.dep/time
            avg_users[runs,buff_index,arrival_index]=data.ut/time
            avg_delay[runs,buff_index,arrival_index]=data.delay/data.dep
            load[runs,buff_index,arrival_index]=SERVICE/a
            
            if IS_BUFFER_UNIQUE==0:
                for s in range(SERVERS_NUMBER):
                    buff_occupance, total_event=servers[s].getBufferOccupanceInfo()
                    plot_avg_buffer_occupation[s,arrival_index,buff_index]+=buff_occupance/total_event/RUNS_NUMBER

                
            elif IS_BUFFER_UNIQUE==1:
                plot_avg_buffer_occupation[0,arrival_index,buff_index]+=total_buffer_occupance/total_event/RUNS_NUMBER
            plot_total_arrivals[arrival_index,buff_index]+=data.arr/RUNS_NUMBER
            plot_total_losses[arrival_index,buff_index]+=losses/RUNS_NUMBER
            plot_total_loss_prob[arrival_index,buff_index]+=(losses/data.arr)/RUNS_NUMBER
            plot_departures[arrival_index,buff_index]+=data.dep/RUNS_NUMBER
            plot_arrival_rate[arrival_index,buff_index]+=data.arr/time/RUNS_NUMBER
            plot_departure_rate[arrival_index,buff_index]+=data.dep/time/RUNS_NUMBER
            plot_avg_users[arrival_index,buff_index]+=data.ut/time/RUNS_NUMBER
            plot_avg_delay[arrival_index,buff_index]+=data.delay/data.dep/RUNS_NUMBER
            if b!=0 and users_in_queue>0:
                if IS_BUFFER_UNIQUE==1:
                    plot_avg_delay_in_queue[arrival_index,buff_index,0]+=data.delay_in_queue/users_in_queue/RUNS_NUMBER
                elif IS_BUFFER_UNIQUE==0:
                    for i in range(BUFFERS_NUMBER):
        
                        #buff_occupance, non_used=servers[i].getBufferOccupanceInfo()
                        buff_queue=servers[i].getBufferUsers()
                        if servers[i].getDelayInQueue()!=0:
                            
                            #print(f'server {servers[i]} had {buff_occupance} users in the queue for a total of {servers[i].getDelayInQueue()}\n')
                            plot_avg_delay_in_queue[arrival_index,buff_index,i]+=servers[i].getDelayInQueue()/buff_queue/RUNS_NUMBER                                
                            
                            #print("\n\n",f"{servers[i].getDelayInQueue()}, {buff_occupance}, {i}, run={runs}, arrival={1/a}, buffer={b}", "\n\n")
                            
                            #print(f'\n\n{plot_avg_delay_in_queue[arrival_index,buff_index,i]}\n\n')
            plot_load[arrival_index,buff_index]+=(SERVICE/a)/RUNS_NUMBER
            
            seed+=1
        
        #confidence interval computation on loss probability for each arrival
        v=0
        loss_prob_mean[arrival_index]=plot_total_loss_prob[arrival_index,CI_BUFF_SIZE_index] 
        for i in range(RUNS_NUMBER):
            v+=(total_loss_prob[i,CI_BUFF_SIZE_index,arrival_index]-loss_prob_mean[arrival_index])**2
            
        variance=v/(RUNS_NUMBER-1)

        lower_bound[arrival_index]=loss_prob_mean[arrival_index]-TS*m.sqrt(variance)/m.sqrt(RUNS_NUMBER)
        upper_bound[arrival_index]=loss_prob_mean[arrival_index]+TS*m.sqrt(variance)/m.sqrt(RUNS_NUMBER)
                
        for s in range(len(servers)):
            myFile.write('Avergae busy time for server'+ str(s) +'with buffer size: '+str(b) +', '+str(busytime[s]/RUNS_NUMBER))
            busy_time_plot[arrival_index, s, buff_index]=(busytime[s]/RUNS_NUMBER)
        arrival_index+=1
        
    plot_all_graphs(ARRIVALS, buff_index, plot_total_arrivals, plot_total_losses, plot_departures, plot_arrival_rate, plot_departure_rate, plot_avg_users, plot_avg_delay, plot_avg_delay_in_queue, plot_total_loss_prob, plot_load, plot_avg_buffer_occupation, b, load_vect, busy_time_plot)
    buff_index+=1
    
#PLOT GRAPH WITH CONFIDENCE INTERVALS
plt.figure(figsize=(6,4))
nn=np.arange(len(ARRIVALS))
plt.xticks(nn, np.round(1/ARRIVALS,3), rotation=90)
plt.xlabel("arrival rate")
plt.ylabel('loss probability')
plt.title('Confidence Interval')
for a in range(ARRIVAL_RATE_NUMBER):
    plot_confidence_interval((ARRIVALS[a]), loss_prob_mean[a], upper_bound[a], lower_bound[a])
plt.grid()
plt.tight_layout()
plt.show()   
#CONFIDENCE INTERVAL ON LOSS PROBABILITY
v=0
loss_prob_mean=plot_total_loss_prob[CI_ARRIVAL_RATE_index,CI_BUFF_SIZE_index] 
for i in range(RUNS_NUMBER):
    v+=(total_loss_prob[i,CI_BUFF_SIZE_index,CI_ARRIVAL_RATE_index]-loss_prob_mean)**2
    
variance=v/(RUNS_NUMBER-1)

lower_bound=loss_prob_mean-TS*m.sqrt(variance)/m.sqrt(RUNS_NUMBER)
upper_bound=loss_prob_mean+TS*m.sqrt(variance)/m.sqrt(RUNS_NUMBER)

myFile.write(f"\n\nCI COMPUTATION WITH ARRIVAL RATE={1/ARRIVALS[CI_ARRIVAL_RATE_index]} AND {RUNS_NUMBER} RUNS\n")
myFile.write(f"MEAN LOSS PROB: {loss_prob_mean}\n")
myFile.write(f"CI UPPER BOUND: {upper_bound}\n")
myFile.write(f"CI LOWER BOUND: {lower_bound}\n")

print(f"\n\nCI COMPUTATION WITH ARRIVAL RATE={1/ARRIVALS[CI_ARRIVAL_RATE_index]} AND {RUNS_NUMBER} RUNS\n")
print(f"MEAN LOSS PROB: {loss_prob_mean}\n")
print(f"CI UPPER BOUND: {upper_bound}\n")
print(f"CI LOWER BOUND: {lower_bound}\n")




myFile.close()

