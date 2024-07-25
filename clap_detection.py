import cv2
import mediapipe as mp
import time
import math
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection

# For webcam input:
cap = cv2.VideoCapture(0)  # 0 : 내장카메라, url : 카메라 주소
pTime = 0
fps_list = deque(maxlen=30)  # 최근 30개의 FPS 값을 저장할 덱
cnt = 0  # 카운트
clap_started = False  # 박수가 시작되었는지 여부

with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands, \
        mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image_height, image_width, _ = image.shape
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to detect hands
        results_hands = hands.process(image)
        # Process the image to detect faces
        results_face = face_detection.process(image)

        # 모자이크 처리
        if results_face.detections:
            for detection in results_face.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # 경계값 조정
                x = max(0, x)
                y = max(0, y)
                w = min(iw - x, w)
                h = min(ih - y, h)

                if w > 0 and h > 0:
                    # 얼굴 영역을 모자이크 처리
                    face_region = image[y:y + h, x:x + w]
                    if face_region.size > 0:  # face_region이 비어있지 않은지 확인
                        blurred_face_region = cv2.resize(face_region, (10, 10), interpolation=cv2.INTER_LINEAR)
                        blurred_face_region = cv2.resize(blurred_face_region, (w, h), interpolation=cv2.INTER_NEAREST)

                        # 사이즈 조정된 얼굴 영역을 원본 이미지에 적용
                        image[y:y + h, x:x + w] = blurred_face_region

        left_hand_index_finger = None
        right_hand_index_finger = None

        # 손 랜드마크 처리
        if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
            hand_landmarks_list = results_hands.multi_hand_landmarks
            handedness_list = results_hands.multi_handedness

            for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
                hand_label = handedness.classification[0].label

                if hand_label == "Left":
                    left_hand_index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                elif hand_label == "Right":
                    right_hand_index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            if left_hand_index_finger and right_hand_index_finger:
                # 거리 계산
                distance = math.sqrt(
                    (left_hand_index_finger.x - right_hand_index_finger.x) ** 2 +
                    (left_hand_index_finger.y - right_hand_index_finger.y) ** 2 +
                    (left_hand_index_finger.z - right_hand_index_finger.z) ** 2
                )

                if distance < 0.05:
                    if not clap_started:
                        cnt += 1
                        print(f'{cnt}회 검지 만남!')
                        clap_started = True
                else:
                    clap_started = False

        # Frame rate 계산
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        fps_list.append(fps)
        avg_fps = sum(fps_list) / len(fps_list)
        cv2.putText(image, f'FPS: {int(avg_fps)}', (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f'Count: {cnt}', (20, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands and Face', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
