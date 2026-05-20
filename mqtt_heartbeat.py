import ssl
import time
import threading
import paho.mqtt.client as mqtt


class MqttHeartbeat(threading.Thread):
    def __init__(
        self,
        broker,
        port,
        username,
        password,
        topic,
        interval_sec=2.0,
        tls_insecure=False,
        reset_topic=None,
        on_reset=None,
        esp32_issue_topic=None,
    ):
        super().__init__(daemon=True)
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.topic = topic
        self.interval_sec = interval_sec
        self.tls_insecure = tls_insecure
        self.reset_topic = reset_topic
        self.on_reset = on_reset
        self.esp32_issue_topic = esp32_issue_topic

        self._stop_event = threading.Event()

        self.client = mqtt.Client(client_id=f"{topic.replace('/', '_')}_hb")
        self.client.username_pw_set(self.username, self.password)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        # TLS for 8883
        self.client.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)
        if self.tls_insecure:
            self.client.tls_insecure_set(True)

        # Optional: “offline” if unexpected disconnect (remove if backend rejects it)
        # self.client.will_set(self.topic, payload="off", qos=0, retain=False)

        self.client.reconnect_delay_set(min_delay=1, max_delay=10)

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0 and self.reset_topic:
            client.subscribe(self.reset_topic, qos=0)
            print(f"✅ MQTT subscribed: {self.reset_topic}")

    def _on_message(self, client, userdata, msg):
        if not self.reset_topic or msg.topic != self.reset_topic:
            return

        payload = msg.payload.decode("utf-8", errors="ignore").strip().lower()
        if payload != "reset":
            return

        print(f"🔁 MQTT reset command received on {self.reset_topic}")
        if self.on_reset:
            try:
                self.on_reset()
            except Exception as exc:
                print(f"⚠️ Reset callback failed: {exc}")

    def publish_reset_success(self):
        if not self.reset_topic:
            return
        self.client.publish(self.reset_topic, payload="reset_success", qos=0, retain=False)
    
    def publish_esp32_issue(self):
        if not self.esp32_issue_topic:
            return
        self.client.publish(self.esp32_issue_topic, payload="issue", qos=0, retain=False)

    def run(self):
        self.client.connect(self.broker, self.port, keepalive=30)
        self.client.loop_start()
        time.sleep(0.5)

        try:
            while not self._stop_event.is_set():
                self.client.publish(self.topic, payload="on", qos=0, retain=False)
                time.sleep(self.interval_sec)
                print(f"# MQTT heartbeat sent at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        except Exception as exc:
            print(f"### MQTT heartbeat error: {exc} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        finally:
            self.client.loop_stop()
            self.client.disconnect()

    def stop(self):
        self._stop_event.set()

