import json
from MyMQTT import *
import time
import cherrypy
from datetime import datetime


class SensorSubscriber(): #THIS PROGRAM RECEIVES DATA VIA MQTT FROM THE SENSORS AND ACTS AS A SERVER FOR PROVIDING INFORMATION TO THE APPLICATIONS

  exposed=True
  def __init__(self,baseTopic,broker,port):
    self.clientID="sensor_sub1"
    self.baseTopic=baseTopic
    self.client=MyMQTT(self.clientID,broker,port, self)
    self.register=[]

  def GET(self, *uri, **parameters): #METHOD FOR HANDLING THE REQUEST FROM APPLICATIONS
    if len(parameters)>=2: 
        print("TELEGRAM GET REQUEST RECEIVED!\n")
        print(self.register)
        print("\n\n")
        
        for entry in self.register:
          print(entry['e'][0]["room"],entry["e"][0]["type"],parameters["room_name"], parameters["sensor_type"], entry["e"][0]["owner"] )
          if entry['e'][0]["room"]==parameters["room_name"] and entry["e"][0]["type"]==parameters["sensor_type"] and parameters["check"]=="value":  #modifica fatta per il telegram warning
              output=entry["e"][0]["value"]
              print("MESSAGE SENT!\n")
              return output
          
          elif entry['e'][0]["room"]==parameters["room_name"] and entry["e"][0]["type"]==parameters["sensor_type"] and parameters["check"]=="all" and entry["e"][0]["owner"]==parameters["owner"]:

              output=entry["e"][0]["value"]+' '+entry["e"][0]["unit"]+' '+ str(datetime.utcfromtimestamp(int(round(float(entry["e"][0]["timestamp"]),0))))+' UTC' 
              print("MESSAGE SENT!\n")
              return output
        
        print("MESSAGE SENT!\n")
        return "NOT FOUND"
    else:
        error_string="incorrect URI"+len(uri)
        raise cherrypy.HTTPError(400, error_string)

  def run(self):
    self.client.start()
    print('{} has started'.format(self.clientID))

  def end(self):
    self.client.stop()
    print('{} has stopped'.format(self.clientID))

  def follow(self,topic):    
    self.client.mySubscribe(topic)
    
  def notify(self, topic, msg):
    payload=json.loads(msg)   
    result=payload
    result_dict=json.loads(result)
    
    flag=0
    for entry in self.register:
        if entry["e"][0]['type']==result_dict["e"][0]['type'] and entry["e"][0]["room"]==result_dict["e"][0]['room'] and entry["e"][0]['owner']==result_dict["e"][0]['owner']:
            entry["e"][0]['value']=result_dict["e"][0]['value']
            entry["e"][0]['owner']=result_dict["e"][0]['owner']
            entry["e"][0]['timestamp']=result_dict["e"][0]['timestamp']
            entry["e"][0]["room"]=result_dict["e"][0]["room"]
            flag=1
    if flag==0:
      self.register.append(result_dict)
 

if __name__ == '__main__':

  config=json.load(open('sensor_sub_conf.json'))
  service_info=json.load(open("service_catalog_info.json"))
  coll=SensorSubscriber(service_info["base_topic"], config["broker"], config["broker_port"])

  conf={
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.sessions.on': True, 
        }
    }
  cherrypy.tree.mount(coll,'/', conf)
  cherrypy.config.update(conf)
  cherrypy.config.update({"server.socket_host":config["ip"]})
  cherrypy.config.update({"server.socket_port":config["port"]})
  cherrypy.engine.start()

  
  coll.run()
  coll.client.unsubscribe()
  result=coll.follow(coll.baseTopic+'/#')
  cherrypy.engine.block()
  coll.end()