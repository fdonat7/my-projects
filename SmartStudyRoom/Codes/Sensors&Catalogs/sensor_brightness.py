from MyMQTT import *
import time
import json
import random
from simplePublisher import *
import requests
import RPi.GPIO as rpi
from gpiozero import LED
import sys


class SensorControl(MyPublisher):
    
    def __init__(self, clientID, sensortype, sensorID, measure, broker, port, topic):
        self.clientID=clientID
        self.sensortype=sensortype
        self.sensorID=sensorID
        self.measure=measure
        self.topic=topic
        self.client=MyMQTT(self.sensorID,broker,port,None)

        self.__message={
                            'bn': self.topic,
                            'e': [
                                    {
                                        'type':self.sensortype,
                                        'unit':self.measure,
                                        'timestamp':'',
                                        'value':' ',
                                        'owner':'',
                                        'room':''
                                    }
                                ]
        }
                
    def start (self):
        self.client.start()
    
    def stop (self):
        self.client.stop()

    def publish(self,value, owner, room):
        message=self.__message
        message['e'][0]['value']=value
        message['e'][0]['timestamp']=str(time.time())
        message['e'][0]['owner']=owner
        message['e'][0]['room']=room
        self.client.myPublish(self.topic,json.dumps(message))
        print("published\n" + json.dumps(message))

def registration(setting_file, service_file):  #IN ORDER TO REGISTER ON THE RESOURCE CATALOG
    with open(setting_file,"r") as f1:    
        conf=json.loads(f1.read())

    with open(service_file,"r") as f2:    
        conf_service=json.loads(f2.read())
    requeststring='http://'+conf_service['ip_address_service']+':'+conf_service['ip_port_service']+'/rooms_name_owner'
    r=requests.get(requeststring)
    print("INFORMATION FROM SERVICE CATALOG RECEIVED!\n")
    print(r.text)
    print("Available rooms and owners:\n "+r.text+"\n")
    owner=input("Which owner? ")
    room=input("\nWhich room? ")

    requeststring='http://'+conf_service['ip_address_service']+':'+conf_service['ip_port_service']+'/room_info?room='+room+'&owner='+owner
    r=requests.get(requeststring)
    print("INFORMATION OF RESOURCE CATALOG (room) RECEIVED!\n") #PRINT FOR DEMO
    
    rc=json.loads(r.text)
    rc_ip=rc["ip_address"]
    rc_port=rc["ip_port"]
    poststring='http://'+rc_ip+':'+rc_port+'/device'
    rc_basetopic=rc["base_topic"]
    rc_broker=rc["broker"]
    rc_port=rc["broker_port"]
    rc_owner=rc["owner"]
    
    sensor_model=conf["sensor_model"]

    requeststring='http://'+conf_service['ip_address_service']+':'+conf_service['ip_port_service']+'/base_topic'
    sbt=requests.get(requeststring)

    service_b_t=json.loads(sbt.text)
    topic=[]
    body=[]
    index=0
    for i in conf["sensor_type"]:
        print(i)
        topic.append(service_b_t+'/'+rc_owner+'/'+rc_basetopic+"/"+i+"/"+conf["sensor_id"])
        body_dic= {
        "sensor_id":conf['sensor_id'],
        "sensor_type":conf['sensor_type'],
        "owner":rc["owner"],
            "measure":conf["measure"][index],
            "end-points":{
                "basetopic":service_b_t+'/'+rc_owner+'/'+rc_basetopic,
                "complete_topic":topic,
                "broker":rc["broker"],
                "port":rc["broker_port"]
            }
        }
        body.append(body_dic)
        requests.post(poststring,json.dumps(body[index]))
        print("REGISTRATION TO RESOURCE CATALOG (room) DONE!\n") #PRINT FOR DEMO
    
        index=index+1

    return rc_basetopic, conf["sensor_type"], conf["sensor_id"], topic, conf["measure"], rc_broker, rc_port, sensor_model, rc["owner"]
    
if __name__ == "__main__":
    clientID, sensortype, sensorID, topic, measure, broker, port, sensor_model, owner=registration(sys.argv[1],"service_catalog_info.json")
    index=0
    Sensor=[]
    for i in sensortype:
        Sensor.append(SensorControl(clientID, i, sensorID, measure[index], broker, port, topic[index]))
        Sensor[index].start()
        index=index+1

    conf=json.load(open("sensor_b_1.json"))

    pinB=conf["pin"]
    rpi.setmode(rpi.BCM)
    rpi.setup(pinB, rpi.IN)
    
    old_result=0
    while 1:
        
        time.sleep(1)    
        result={'result':rpi.input(pinB)}
        print(result)
        poststring='http://'+conf["server_ip"]+':'+str(conf["server_port"])
        requests.post(poststring,json.dumps(result)) #POSTING INFORMATION TO BRIGHTNESS CONTROL SERVER
        if result is not None:
            print(f"result= {result}")
            Sensor[0].publish(result, owner, clientID)
        else:
            print('Failed to get reading. Try again!')
        
