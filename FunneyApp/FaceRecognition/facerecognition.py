import cv2
import face_recognition
import os
import datetime
from PIL import Image

def save_face_image(face_image, save_dir="faces"):
    # 检查保存目录是否存在，不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 使用时间戳生成唯一文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"face_{timestamp}.jpg"
    filepath = os.path.join(save_dir, filename)
    # 保存人脸图片到指定路径
    cv2.imwrite(filepath, face_image)
    print(f"人脸照片已保存: {filepath}")

def main():
    cap = cv2.VideoCapture(0)
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开本地摄像头。")
        return
    print("按 's' 键保存检测到的人脸，按 'q' 键退出。")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头。")
            break
        #转换为RGB格式以供face_recognition使用
        # 将摄像头获取的图像转换为8bit RGB图像
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # rgb_frame 类型为8bit RGB图像，可用于face_recognition
        #print(rgb_frame)
        face_locations = face_recognition.face_locations(rgb_frame)

        # 只在检测到人脸时绘制矩形
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and face_locations:
            for (top, right, bottom, left) in face_locations:
                # 检查坐标合法性，避免保存空图片
                if top < bottom and left < right:
                    face_image = frame[top:bottom, left:right]
                    save_face_image(face_image)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
