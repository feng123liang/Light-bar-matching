import time
import random
from paho.mqtt import client as mqtt_client
 
broker = '10.108.4.125'
port = 1883
topic = "default"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'
 
 
def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)
 
    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION1,client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client
 
 
def publish(client):
    msg_count = 0
    while True:
        time.sleep(1)
        
        msg = {'pumpid' : 0, 
               'time': "00:00:00",
               'predict_status' : 'health' ,
               'time_domian_data' : {[],[],[],[],[],[]} , 
               'fft_domian_data' : [],
               }
        result = client.publish(topic, msg)


        # result: [0, 1]
        status = result[0]
        if status == 0:
            print(f"Send `{msg}` to topic `{topic}`")
        else:
            print(f"Failed to send message to topic {topic}")
        msg_count += 1
 
 
def run():
    client = connect_mqtt()
    client.loop_start()
    publish(client)
 
 
if __name__ == '__main__':
    run()