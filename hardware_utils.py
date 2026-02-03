import serial.tools.list_ports
import cv2


def find_esp32():
    for p in serial.tools.list_ports.comports():
        # Match by USB VID/PID
        if p.vid == 0x303A and p.pid == 0x1001:
            return p.device
    return None
            

def find_camera():
    cam_list=["/dev/video0","/dev/video1","/dev/video2"]
    for cam in cam_list:
        cap = cv2.VideoCapture(cam)
        if cap.isOpened():
            cap.release()
            return cam
    return cam_list[0]


print(find_esp32())
print(find_camera())

