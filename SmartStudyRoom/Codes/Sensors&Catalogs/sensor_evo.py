from MyMQTT import *
import time
import json
import random
from simplePublisher import *
import requests
from smartcard.System import readers
import array
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
        print("Published!\n" + json.dumps(message))

    def read(self): #READING THE FISCAL CODE FROM EVO
        fiscal_code=''
        try: 
            r = readers()
            

            reader = r[0]

            connection = reader.createConnection()
            connection.connect() 

            SELECT_MF = [0x00, 0xA4, 0x00, 0x00, 0x02, 0x3F, 0x00]
            data, sw1, sw2 = connection.transmit(SELECT_MF)
            

            SELECT_DF1 = [0x00, 0xA4, 0x00, 0x00, 0x02, 0x11, 0x00]
            data, sw1, sw2 = connection.transmit(SELECT_DF1)
            
            SELECT_EF_PERS = [0x00, 0xA4, 0x00, 0x00, 0x02, 0x11, 0x02]
            data, sw1, sw2 = connection.transmit(SELECT_EF_PERS)

            READ_BIN = [0x00, 0xB0, 0x00, 0x00, 0x00, 0x00]
            data, sw1, sw2 = connection.transmit(READ_BIN)
            
            personal_data = array.array('B', data).tobytes()
            
            dimension = int(personal_data[0:6],16)
            
            fromFC = 68
            toFC = 84
            fiscal_code = personal_data[fromFC:toFC]
            print ("\n", "\n", "\n", "CF: ", fiscal_code)
            
        except: 
            pass
        if fiscal_code is not None and fiscal_code != '': 
            return fiscal_code
        return ''

def bookingControl (fiscal_code): #RECEIVING BOOKINGS INFORMATION FROM STUDENTS DATABASE AND CHECKING IDENTITY
    conf=json.load(open("sensor_evo_1.json"))       
    port=conf["server_port"]
    string='http://'+conf["server_ip"]+':'+str(conf["server_port"])+'/all_bookings'
    r=requests.get(string)
    print("\nBOOKINGS RECEIVED FROM STUDENTS DATABASE!\n") #PRINT FOR DEMO
    fiscal_code_server=json.loads(r.text)

    for entry in fiscal_code_server:
        booked_code=bytes(entry['fiscal_code'],'utf-8')
        if booked_code==fiscal_code:
            return "ACCESS APPROVED!"
    return "ACCESS DENIED!"



def registration(setting_file, service_file): #IN ORDER TO REGISTER ON THE RESOURCE CATALOG
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
    clientID, sensortype, sensorID, topic, measure, broker, port, sensor_model, owner=registration(sys.argv[1], "service_catalog_info.json")
    index=0
    Sensor=[]

    for i in sensortype:
        Sensor.append(SensorControl(clientID, i, sensorID, measure[index], broker, port, topic[index]))
        Sensor[index].start()
        index=index+1

    conf=json.load(open("sensor_evo_1.json"))
    while 1:
        while(1):
            fiscal_code=Sensor[0].read()
            if fiscal_code!='':
                break     
        isbooked=bookingControl(fiscal_code)
        time.sleep(10) 
        Sensor[0].publish(isbooked, owner, clientID)

        poststring='http://'+conf["server_ip"]+':'+str(conf["server_port"])
        requests.post(poststring,json.dumps(isbooked))     #POSTING INFORMATION TO STUDENTS DATABASE SERVER

        print(isbooked)