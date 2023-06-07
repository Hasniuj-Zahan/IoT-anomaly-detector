import csv
import pywemo
import asyncio
from kasa import *
import tinytuya
import time
import os
import lightgovee

def bedroomlight1():
    bulb = SmartBulb("192.168.1.15")
    #asyncio.run(bulb.set_led(False))
    #asyncio.run(bulb.turn_off())
    asyncio.run(bulb.update())
    return bulb.light_state['on_off']

def bedroombulb2():

    return (lightgovee.state(3)['data']['properties'][1]['powerState'])

def livingroomlight1():
    bulb = SmartBulb("192.168.1.5")
    #asyncio.run(bulb.set_led(False))
    #asyncio.run(bulb.turn_off())
    asyncio.run(bulb.update())
    return bulb.light_state['on_off']


def drawingplug1():
    plug = SmartPlug("192.168.1.6")
    #asyncio.run(bulb.set_led(False))
    #asyncio.run(bulb.turn_off())
    asyncio.run(plug.update())
    #x = print(plug.is_on)
    return plug.is_on

def MotionDetection():
    TP = tinytuya.OutletDevice('eb01fded814e43da57xpdc', '192.168.1.13', '4617bac43ae7c5a6')
    TP.set_version(3.3)
    data = TP.status()
    # Toggle switch state
    switch_state = data['dps']['1']
    #x = print('switch:', switch_state)
    return switch_state

def AnotherPlug():
    plug = tinytuya.OutletDevice('ebcb2c36ac63216fbfkujs', '192.168.1.11', 'b5771c60b65172e3')
    plug.set_version(3.3)
    data = plug.status()
    switch_state = data['dps']['1']
    return switch_state

def Multiplug():
    MP = tinytuya.OutletDevice('11766315807d3a5996af', '192.168.1.3', '39c1c903384426a2')
    MP.set_version(3.1)
    data = MP.status()
    # check switch state
    switch_state = data['dps']['3']
    #x = print(switch_state)
    return switch_state


def WemoPlug():
    devices = pywemo.discover_devices()
    # devices[0].toggle()
    #x = print(devices[0].get_state())
    return devices[0].get_state()


def Kettle():
    kettle = tinytuya.OutletDevice('45553020a4e57cad155d', '192.168.1.2', '37ccea4b826aba8e')
    kettle.set_version(3.3)
    data = kettle.status()
    kettlestate = data['dps']['1']
    #watertemp = data ['dps']['5']
    return kettlestate

def Kettletemp():
    kettle = tinytuya.OutletDevice('45553020a4e57cad155d', '192.168.1.2', '37ccea4b826aba8e')
    kettle.set_version(3.3)
    data = kettle.status()
    #kettlestate = data['dps']['1']
    watertemp = data ['dps']['5']
    return watertemp

def Thermostat():
    kettle = tinytuya.OutletDevice('45553020a4e57cad155d', '192.168.1.3', 'f049e6b01289fcd0')
    kettle.set_version(3.3)
    data = kettle.status()

if __name__ == '__main__':
    n = 100

    for i in range(n):
        if not os.path.exists('devicestate' + str(i) + '.csv'):
            with open('devicestate'+str(i)+'.csv', 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['time', 'BedroomLight-1', 'BedroomLight-2', 'LivingroomPlug1', 'MultiPlug', 'MotionDetector', 'KettleTemp', 'Kettle', 'BedrooomPlug', 'BedrooomPlug2', 'Livingroomlight', 'Label'])
                # w.writerow(['Time', 'SourceIP', 'DestinationIP', 'Length', 'DestinationPort', 'Info'])
                txtfilename = ('devicestate' + str(i) + '.csv')



                t_end = time.time() + 60 * 1200
                while time.time() < t_end:
                    A = bedroomlight1()
                    B = bedroombulb2()
                    C = drawingplug1()
                    D = Multiplug()
                    E = MotionDetection()
                    F = Kettletemp()
                    G = Kettle()
                    H = AnotherPlug()
                    I = WemoPlug()
                    J = livingroomlight1()

                    #print(A, B, C, D, E, F, G, H)


                    with open(txtfilename, 'a') as f:
                        f.write(str(time.time()) + "," + str(A) + "," + str(B) + "," + str(C) + "," + str(D) + ',' + str(E) + "," + str(F) + "," + str(G) +"," + str(H)+ "," + str(I)+"," + str(J)+ ',1' + '\n')
            break