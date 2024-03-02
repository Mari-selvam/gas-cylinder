import serial
import time

ser = serial.Serial('/dev/cu.usbmodem14101',9600)
print(ser.name)  

ser.write(b'H')
ser.close()

