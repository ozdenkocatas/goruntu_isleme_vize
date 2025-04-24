import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def koordinat_getir(landmarks, indeks, h, w):
    landmark = landmarks[indeks]
    return int(landmark.x * w), int(landmark.y * h)

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    h, w, _ = annotated_image.shape

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx][0].category_name
        parmaklar = []

        # Diğer 4 parmak: y ekseni kontrolü
        parmaklar.append(1 if koordinat_getir(hand_landmarks, 8, h, w)[1] < koordinat_getir(hand_landmarks, 6, h, w)[1] else 0)
        parmaklar.append(1 if koordinat_getir(hand_landmarks, 12, h, w)[1] < koordinat_getir(hand_landmarks, 10, h, w)[1] else 0)
        parmaklar.append(1 if koordinat_getir(hand_landmarks, 16, h, w)[1] < koordinat_getir(hand_landmarks, 14, h, w)[1] else 0)
        parmaklar.append(1 if koordinat_getir(hand_landmarks, 20, h, w)[1] < koordinat_getir(hand_landmarks, 18, h, w)[1] else 0)

        # Geliştirilmiş baş parmak kontrolü
        x4, y4 = koordinat_getir(hand_landmarks, 4, h, w)
        x2, y2 = koordinat_getir(hand_landmarks, 2, h, w)
        x0, y0 = koordinat_getir(hand_landmarks, 0, h, w)

        d_thumb = np.hypot(x4 - x2, y4 - y2)      # baş parmak boğum mesafesi
        d_wrist = np.hypot(x0 - x2, y0 - y2)      # bilek ile baş parmak boğum mesafesi

        parmaklar.append(1 if d_thumb > d_wrist * 0.7 else 0)

        toplam = sum(parmaklar)
        x_text, y_text = koordinat_getir(hand_landmarks, 0, h, w)

        cv2.putText(annotated_image, f"{handedness}: {toplam}", (x_text, y_text - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

    return annotated_image

# Modeli yükle
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Kamerayı başlat
cam = cv2.VideoCapture(0)
while cam.isOpened():
    basari, frame = cam.read()
    if not basari:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    detection_result = detector.detect(mp_image)

    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    cv2.imshow("Elde Açık Parmak Sayısı", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

cam.release()
cv2.destroyAllWindows()
