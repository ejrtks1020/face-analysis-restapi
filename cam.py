
from unittest import result
import cv2
from deepface import DeepFace

def draw_info(image, face_info_list, race):
    for face in face_info_list:
        x = face['region']['x']
        y = face['region']['y']
        w = face['region']['w']
        h = face['region']['h']
        emotion = face['dominant_emotion']
        gender = face['gender']
        age = face['age']
        if race:
            race = face['dominant_race']
        # print(x, y, w, h)
        if w != image.shape[0]:
            cv2.rectangle(image, (x, y), (x+w, y+h),
                        color=(0, 255, 0), thickness=2)
            cv2.putText(image, f'{emotion}', (int(x + w*1.01), y),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
            cv2.putText(image, f'{gender}', (int(x + w*1.01), int(y + h*0.3)),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
            cv2.putText(image, f'{age}', (int(x + w*1.01), int(y + h*0.6)),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
            if race:
                cv2.putText(image, f'{race}', (int(x + w*1.01), int(y + h*0.9)),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
        else:
            pass

def get_stream_video():
    # camera 정의
    cam = cv2.VideoCapture(0)

    while True:
        # 카메라 값 불러오기
        success, frame = cam.read()
        frame_copy = frame.copy

        if not success:
            break
        else:
            results = DeepFace.analyze(img_path = frame_copy,
                actions=['emotion', 'age', 'gender'],
                detector_backend = 'retinaface',
                enable_multiple=True,
                prog_bar=False
            )
            face_info_list = [face for face in results.values()]
            draw_info(frame_copy, face_info_list, False)
            ret, buffer = cv2.imencode('.jpg', frame)
            # frame을 byte로 변경 후 특정 식??으로 변환 후에
            # yield로 하나씩 넘겨준다.
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(frame) + b'\r\n')