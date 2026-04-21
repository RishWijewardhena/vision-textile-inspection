#!/usr/bin/env python3
"""
Simple MQTT reset test sender.
Publishes `reset` and waits for `reset_success` on the same topic.

Usage:
    python3 mqtt_reset_test.py
"""

import ssl
import sys
import threading
import time

import paho.mqtt.client as mqtt

# Hardcoded test config (no .env)
BROKER = "mqtt.anc.idea8.cloud"
PORT = 8883
USERNAME = "backend"
PASSWORD = "bbf12cwcpm"
TLS_INSECURE = True
DEVICE_ID = "line_01"  # Set this to your backend DB_TABLE value
TIMEOUT_SEC = 8.0


class ResetTester:
    def __init__(self):
        self.topic = f"machine/{DEVICE_ID}/commands/reset"
        self.timeout_sec = TIMEOUT_SEC
        self.ack_event = threading.Event()

        client_id = f"reset_test_{DEVICE_ID}_{int(time.time())}"
        self.client = mqtt.Client(client_id=client_id)
        self.client.username_pw_set(USERNAME, PASSWORD)
        self.client.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)
        if TLS_INSECURE:
            self.client.tls_insecure_set(True)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        if rc != 0:
            print(f"Connect failed with rc={rc}")
            return
        print(f"Connected. Subscribing to: {self.topic}")
        client.subscribe(self.topic, qos=0)
        print("Publishing: reset")
        client.publish(self.topic, payload="reset", qos=0, retain=False)

    def on_message(self, client, userdata, msg):
        payload = msg.payload.decode("utf-8", errors="ignore").strip().lower()
        print(f"Received on {msg.topic}: {payload}")
        if payload == "reset_success":
            self.ack_event.set()

    def run(self) -> int:
        self.client.connect(BROKER, PORT, keepalive=30)
        self.client.loop_start()
        try:
            ok = self.ack_event.wait(timeout=self.timeout_sec)
            if ok:
                print("PASS: reset_success received")
                return 0
            print("FAIL: timeout waiting for reset_success")
            return 1
        finally:
            self.client.loop_stop()
            self.client.disconnect()

def main():
    tester = ResetTester()
    print(f"Using topic: machine/{DEVICE_ID}/commands/reset")
    return tester.run()


if __name__ == "__main__":
    sys.exit(main())
