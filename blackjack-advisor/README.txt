Team Members: [Your Name], [Partner's Name]

Project: Real-Time Blackjack Card Counting Advisor
Course:  EE250 - Distributed Systems for the Internet of Things, Spring 2026

============================================================
SYSTEM OVERVIEW
============================================================
An end-to-end IoT system that uses a Raspberry Pi with an overhead camera
to automatically detect playing cards in real time, run Hi-Lo card counting
logic, compute the expected value (EV) of every possible blackjack action,
and deliver live recommendations via a local web dashboard and SMS.

Node 1: Raspberry Pi 4 (camera capture, CV pipeline, CNN inference,
         card counting, EV calculation, MQTT publishing, SQLite logging)
Node 2: Cloud services — HiveMQ Cloud (MQTT broker), InfluxDB Cloud
         (time-series data), Grafana Cloud (dashboard), Twilio (SMS)

============================================================
HOW TO RUN
============================================================
1. SSH into the Raspberry Pi:
       ssh pi@blackjackpi.local

2. Navigate to the project directory:
       cd blackjack-advisor/rpi

3. Ensure credentials are filled in:
       - mqtt_publisher.py  (HiveMQ Cloud broker hostname, username, password)
       - sms_sender.py      (Twilio Account SID, Auth Token, phone numbers)
       - logger.py          (InfluxDB Cloud URL, token, org name)

4. Place the trained model in the rpi/ folder:
       model.tflite

5. Run the main script:
       python3 main.py

6. Open the web dashboard from any device on the same network:
       http://blackjackpi.local:5000

7. Point the overhead camera at the card table.
   Deal cards normally — they will be detected automatically.

============================================================
EXTERNAL LIBRARIES USED
============================================================
- picamera2            RPi Camera Module 3 capture (IMX708)
- opencv-python-headless  Computer vision pipeline (card detection)
- tflite-runtime       TensorFlow Lite model inference on RPi
- paho-mqtt            MQTT publish/subscribe (Lab 4)
- flask                Web server and local dashboard UI (Lab 3)
- requests             HTTP/REST API calls to Twilio (Lab 3)
- twilio               Twilio SMS REST API wrapper
- influxdb-client      InfluxDB Cloud time-series logging (Lab 9)
- sqlite3              Local session logging (Python built-in)
- numpy                Array operations for image processing

============================================================
TRAINING THE CNN MODEL
============================================================
See training/train.ipynb (run on Google Colab with GPU runtime).
After training, export model.tflite and copy it to rpi/model.tflite.

============================================================
AI TOOL ACKNOWLEDGMENT
============================================================
AI tools (Claude by Anthropic) were used to assist with code development,
architecture planning, and documentation. All design decisions are our own
and we can explain every component of the system during the demo.
