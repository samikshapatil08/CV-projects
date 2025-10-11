import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

def get_landmark_coords(landmarks, landmark_index, img_shape):
    h, w, _ = img_shape
    lm = landmarks.landmark[landmark_index]
    return (lm.x * w, lm.y * h)

def detect_emotion(face_landmarks, img_shape):
    upper_lip = get_landmark_coords(face_landmarks, 13, img_shape)
    lower_lip = get_landmark_coords(face_landmarks, 14, img_shape)
    left_mouth_corner = get_landmark_coords(face_landmarks, 61, img_shape)
    right_mouth_corner = get_landmark_coords(face_landmarks, 291, img_shape)

    left_eye_inner = get_landmark_coords(face_landmarks, 133, img_shape)
    right_eye_inner = get_landmark_coords(face_landmarks, 362, img_shape)
    inter_ocular_distance = np.linalg.norm(np.array(left_eye_inner) - np.array(right_eye_inner))

    if inter_ocular_distance == 0:
        return "Neutral"

    mouth_openness = np.linalg.norm(np.array(upper_lip) - np.array(lower_lip)) / inter_ocular_distance
    mouth_width = np.linalg.norm(np.array(left_mouth_corner) - np.array(right_mouth_corner)) / inter_ocular_distance

    left_eyebrow_top = get_landmark_coords(face_landmarks, 105, img_shape)
    left_eye_top = get_landmark_coords(face_landmarks, 159, img_shape)
    eyebrow_raise = (left_eye_top[1] - left_eyebrow_top[1]) / inter_ocular_distance
    if mouth_openness > 0.4 and eyebrow_raise > 0.2:
        return "Surprise"

    left_eyelid_upper = get_landmark_coords(face_landmarks, 159, img_shape)
    left_eyelid_lower = get_landmark_coords(face_landmarks, 145, img_shape)
    left_eye_openness = np.linalg.norm(np.array(left_eyelid_upper) - np.array(left_eyelid_lower)) / inter_ocular_distance
    if left_eye_openness > 0.2 and mouth_openness > 0.3:
        return "Fear"

    upper_lip_y = get_landmark_coords(face_landmarks, 13, img_shape)[1]
    nose_tip_y = get_landmark_coords(face_landmarks, 4, img_shape)[1]
    disgust_score = (nose_tip_y - upper_lip_y) / inter_ocular_distance
    if disgust_score > 0.15 and mouth_openness < 0.1:
        return "Disgust"

    left_inner_eyebrow = get_landmark_coords(face_landmarks, 55, img_shape)
    right_inner_eyebrow = get_landmark_coords(face_landmarks, 285, img_shape)
    eyebrow_furrow = np.linalg.norm(np.array(left_inner_eyebrow) - np.array(right_inner_eyebrow)) / inter_ocular_distance
    mouth_corners_y = (left_mouth_corner[1] + right_mouth_corner[1]) / 2
    mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2
    mouth_downturn = (mouth_corners_y - mouth_center_y) / inter_ocular_distance
    if eyebrow_furrow < 0.3 and mouth_downturn > 0.05:
        return "Anger"

    smile_score = (mouth_center_y - mouth_corners_y) / inter_ocular_distance
    if smile_score > 0.05:
        return "Happiness"
    if smile_score < -0.05:
        return "Sadness"

    return "Neutral"

def main():
    image_path = "your_image.jpg"
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error: Could not read the image from {image_path}")
        return

    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

        cv2.putText(frame, f"Emotion: {emotion}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Emotion and Pose Detection", frame)
        cv2.waitKey(0)
    finally:
        pose.close()
        face_mesh.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
