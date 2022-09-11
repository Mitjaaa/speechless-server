import time
import socket
import os
from _thread import *

import numpy as np
import sounddevice
import stt
import pyaudio
import webrtcvad

ServerSocket = socket.socket()
host = '192.168.2.118'
port = 4044

ThreadCount = 0
clients = []
try:
    ServerSocket.bind((host, port))
except socket.error as e:
    print(str(e))

print('Waiting for a Connection..')
ServerSocket.listen(5)

data_list = []

curr_data = None
is_speech = False

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                output=True)

stream_context = None
send_text = ""
can_send = False


def threaded_client(connection):
    global curr_data, send_text, can_send
    is_host = False
    while True:
        data = connection.recv(2048)
        if not data:
            continue

        if data and data.decode("utf-8", errors="ignore") == "##host##":
            is_host = True
            start_new_thread(classify, ())

        if is_host:
            data_list.append(data)
            curr_data = data

            # vad
            handle_vad(curr_data)

        if len(send_text) > 0 and can_send:
            for conn in clients:
                print(send_text)
                conn.send(str.encode(send_text))
            send_text = ""
            can_send = False

    connection.close()


# Load DeepSpeech model
model_dir = "./"
model_path = os.path.join(model_dir, 'model.tflite')
scorer_path = os.path.join(model_dir, 'scorer.scorer')

model = stt.Model(model_path)
model.enableExternalScorer(scorer_path)

vad = webrtcvad.Vad(3)


def handle_vad(frame):
    global is_speech

    if len(frame) != 640 or frame is None:
        return

    is_speech = vad.is_speech(frame, 16000)
    if not is_speech:
        data_list.clear()


def classify():
    count = 0
    while True:
        global stream_context, send_text, can_send
        stream_context = model.createStream()
        if is_speech and len(data_list) > 0:
            for frame in data_list:
                if frame is not None:
                    stream_context.feedAudioContent(np.frombuffer(frame, np.int16))

            text = stream_context.finishStream()
            #print("Recognized: %s" % text)
            if len(text) > 0:
                if len(text) > len(send_text) or len(text) == len(send_text):
                    send_text = text
                    count += 1
            elif count > 4:
                can_send = True
                count = 0
        elif count > 4:
            can_send = True
            count = 0

            stream_context = model.createStream()


while True:
    Client, address = ServerSocket.accept()
    print('Connected to: ' + address[0] + ':' + str(address[1]))

    clients.append(Client)

    start_new_thread(threaded_client, (Client,))
    ThreadCount += 1
    print('Thread Number: ' + str(ThreadCount))

    time.sleep(5)

    print("Listening (ctrl-C to exit)...")


ServerSocket.close()
