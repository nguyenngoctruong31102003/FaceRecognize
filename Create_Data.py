import cv2
import os

# Thiết lập camera với các thông số weight, height
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# Mô hình nhận dạng được train sẵn của Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Khởi tạo Face_ID cho từng khuôn mặt
# ID là bắt đầu từ 0
# Ví dụ có 2 người thì chạy file này 2 lần rồi nhập ID tương ứng
ID = input('\n Nhập ID khuôn mặt  ==>  ')

print('\n [INFO] Khởi tạo Camera ...')
count = 0

while 1:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        count += 1
        # lưu ảnh đã chụp vào Folder Data với dạng jpg
        # Khi lưu ảnh sẽ có dạng sau: 'User.1.1.ipg' với số 1 đầu ứng với ID, 
        # và tiếp theo là số thứ tự của ảnh
        cv2.imwrite('Data/User.' + str(ID) + '.' + str(count) + '.jpg', gray[y:y+h, x:x+w])
        cv2.imshow('Image', img)
                
    
    cv2.imshow('Face Detection', img)
    k = cv2.waitKey(100) & 0xff
    if k == 27: break
    elif count >= 30: break   #Lấy data hình ảnh khuôn mặt với 30 ảnh của 1 người

print('\n Đã chụp hoàn tất!')
print('\n [INFO] Thoat')
cam.release()
cv2.destroyAllWindows()