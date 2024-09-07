import time
import random
import json
from paho.mqtt import client as mqtt_client

class MQTTClient:
    def __init__(self, broker='10.5.153.251', port=1883, topic="default", id = 0):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client_id = f'pump-mqtt-{id}'
        self.client = self.connect_mqtt()

    def connect_mqtt(self):
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT Broker!")
            else:
                print("Failed to connect, return code %d\n", rc)

        try:
            client = mqtt_client.Client(self.client_id)
        except:
            client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION1, self.client_id)

        client.on_connect = on_connect
        client.connect(self.broker, self.port)
        return client

    def publish(self):
        msg_count = 0
        while True:
            time.sleep(1)

            msg = {'pumpid': 0,
                   'time': "00:00:00",
                   'predict_status': 'health',
                   'time_domain_data': [[], [], [], [], [], []],
                   'fft_domain_data': []
                   }
            msg_str = json.dumps(msg)
            result = self.client.publish(self.topic, msg_str)

            status = result[0]
            if status == 0:
                print(f"Send `{msg_str}` to topic `{self.topic}`")
            else:
                print(f"Failed to send message to topic {self.topic}")
            msg_count += 1

    def my_publish(self,pumpid,time,predict_status,time_domain_data,fft_domian_data):
        msg = {'pumpid': pumpid,
                   'time': time,
                   'predict_status': predict_status,
                   'time_domain_data': time_domain_data,
                   'fft_domain_data': fft_domian_data
                }
        msg_str = json.dumps(msg)
        result = self.client.publish(self.topic, msg_str)

        status = result[0]
        if status == 0:
            print(f"Send success to topic `{self.topic}`")
        else:
            print(f"Failed to send message to topic {self.topic}")
        # msg_count += 1

    def run(self):
        self.client.loop_start()
        # self.publish()

# Example of how to use the MQTTClient class
if __name__ == '__main__':
    mqtt_client = MQTTClient()
    mqtt_client.run()