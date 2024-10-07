import cv2
import numpy as np
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Trainer/trainer.yml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['Truong','Huy', 'Obama']

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Tạo thư mục Unknown_Image nếu chưa tồn tại
if not os.path.exists('Unknown_Image'):
    os.makedirs('Unknown_Image')

# Configure email
email_user = 
email_password = 
email_send = 

subject = 'Thông báo: Đã phát hiện người lạ'

def send_email(img_path):
    msg = MIMEMultipart()   # Đóng gói dữ liệu email
    msg['From'] = email_user
    msg['To'] = email_send
    msg['Subject'] = subject

    # Tạo nội dung gửi và ảnh đính kèm
    # ----Begin----
    body = 'Một người lạ đã được phát hiện. Xem hình ảnh đính kèm.'
    msg.attach(MIMEText(body, 'plain'))

    attachment = open(img_path, 'rb')
    part = MIMEBase('application', 'octet-stream') #Kiểu dữ liệu loại tệp không cấu trúc
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(img_path)}")
    
    msg.attach(part)
    text = msg.as_string()
    # ----End----
    
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()   # Bảo mật kết nối với giao thức TLS
        server.login(email_user, email_password)
        server.sendmail(email_user, email_send, text)

# VideoWriter setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Video_ghiHinh/output.avi', fourcc, 50.0, (640, 480))

while 1:
    text = 'Face not Detected'
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH))
    )
    detected = False
    
    for (x, y, w, h) in faces:
        text = 'Face Dectected'
        detected = True
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        if confidence < 100:
            id = names[id]
            confidence_text = "  {0}%".format(round(100 - confidence))
        else:
            id = 'Unknow'
            confidence_text = "  {0}%".format(round(100 - confidence))
            img_path = f'Unknown_Image/unknown_{x}_{y}.jpg'
            cv2.imwrite(img_path, img[y:y+h, x:x+w])
            send_email(img_path)
            
        cv2.putText(img, str(id), (x+5, y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence_text), (x+5, y+h-5), font, 1, (255,255,0), 1)

    if detected:
        out.write(img)
    
    image = cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Nhan Dang Khuon Mat", img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: break
    
cam.release()
out.release()
cv2.destroyAllWindows()
