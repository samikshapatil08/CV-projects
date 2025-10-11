import time
import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

def angle_between_points(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 0.0
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def get_landmark_coords(landmarks, landmark_index, img_shape):
    h, w, _ = img_shape
    lm = landmarks.landmark[landmark_index]
    return (lm.x * w, lm.y * h)

def detect_emotion(face_landmarks, img_shape):
    upper_lip = get_landmark_coords(face_landmarks, 13, img_shape)
    lower_lip = get_landmark_coords(face_landmarks, 14, img_shape)
    left_mouth_corner = get_landmark_coords(face_landmarks, 61, img_shape)
    right_mouth_corner = get_landmark_coords(face_landmarks, 291, img_shape)
    
    mouth_width = np.linalg.norm(np.array(left_mouth_corner) - np.array(right_mouth_corner))
    if mouth_width == 0:
        return "Neutral"

    mouth_openness = np.linalg.norm(np.array(upper_lip) - np.array(lower_lip)) / mouth_width
    left_eyebrow_top = get_landmark_coords(face_landmarks, 105, img_shape)
    left_eye_top = get_landmark_coords(face_landmarks, 159, img_shape)
    eyebrow_raise = (left_eye_top[1] - left_eyebrow_top[1]) / mouth_width
    
    if mouth_openness > 0.4 and eyebrow_raise > 0.1:
        return "Surprise"

    left_eye_inner = get_landmark_coords(face_landmarks, 161, img_shape)
    left_eye_outer = get_landmark_coords(face_landmarks, 154, img_shape)
    left_eye_width = np.linalg.norm(np.array(left_eye_inner) - np.array(left_eye_outer))

    left_eyelid_upper = get_landmark_coords(face_landmarks, 159, img_shape)
    left_eyelid_lower = get_landmark_coords(face_landmarks, 145, img_shape)
    left_eye_openness = np.linalg.norm(np.array(left_eyelid_upper) - np.array(left_eyelid_lower)) / left_eye_width if left_eye_width > 0 else 0
    
    if left_eye_openness > 0.4 and mouth_openness > 0.2:
        return "Fear"

    upper_lip_y = get_landmark_coords(face_landmarks, 13, img_shape)[1]
    nose_tip_y = get_landmark_coords(face_landmarks, 4, img_shape)[1]
    disgust_score = (nose_tip_y - upper_lip_y) / mouth_width
    
    if disgust_score > 0.3:
        return "Disgust"

    left_inner_eyebrow = get_landmark_coords(face_landmarks, 55, img_shape)
    right_inner_eyebrow = get_landmark_coords(face_landmarks, 285, img_shape)
    eyebrow_furrow = np.linalg.norm(np.array(left_inner_eyebrow) - np.array(right_inner_eyebrow)) / mouth_width
    
    if eyebrow_furrow < 0.25:
        return "Anger"
    
    mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2
    mouth_corners_y = (left_mouth_corner[1] + right_mouth_corner[1]) / 2
    smile_score = (mouth_center_y - mouth_corners_y) / mouth_width
    
    if smile_score > 0.08:
        return "Happiness"
    if smile_score < -0.05:
        return "Sadness"

    return "Neutral"

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    prev_time = 0.0
    
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            
            results_pose = pose.process(rgb)
            results_face = face_mesh.process(rgb)

            emotion = "Neutral"

            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(200,200,200), thickness=1, circle_radius=1)
                    )
                    
                    emotion = detect_emotion(face_landmarks, frame.shape)

            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2)
                )

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time else 0.0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Emotion: {emotion}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Emotion and Pose Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
    finally:
        pose.close()
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
