import os
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean as scipy_euclidean
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def quaternion_from_two_vectors(v1, v2):
    cross_product = np.cross(v1, v2)
    w = np.sqrt(np.linalg.norm(v1) * np.linalg.norm(v2)) + np.dot(v1, v2)
    quaternion = np.array([w, cross_product[0], cross_product[1], cross_product[2]])
    return quaternion / np.linalg.norm(quaternion)

def drawArmLeg(annotated_image,joint1,joint2,joint3,color):
    height, width, _ = annotated_image.shape
    
    x0 = int(joint1[0] * width)
    y0 = int(joint1[1] * height)
    cv2.circle(annotated_image, (x0, y0), 5, color, -1)
    
    x = int(joint2[0] * width)
    y = int(joint2[1] * height)
    cv2.circle(annotated_image, (x, y), 5, color, -1)

    x2 = int(joint3[0] * width)
    y2 = int(joint3[1] * height)
    cv2.circle(annotated_image, (x2, y2), 5, color, -1)
    
    cv2.line(annotated_image,(x0,y0),(x, y) ,color, 2)
    cv2.line(annotated_image,(x, y) ,(x2, y2)  ,color, 2)

def collectData(video):
    # 初始化
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video)

    # 用於存儲右手關節的坐標
    shoulder_coordinates = []
    elbow_coordinates = []
    wrist_coordinates = []
    leftShoulder_coordinates = []
    leftElbow_coordinates = []
    leftWrist_coordinates = []

    # 初始化滑動平均變數
    alpha = 0.8  # 平滑因子
    smoothed_shoulder = None
    smoothed_wrist= None
    smoothed_elbow=None

    smoothed_Leftshoulder = None
    smoothed_Leftwrist= None
    smoothed_Leftelbow=None

    # 初始化軌跡
    trajectory_x = []
    trajectory_y = []
    trajectory_z = []

    elbow_wrist_x = []
    elbow_wrist_y = []
    elbow_wrist_z = []
    elbow_angles = []
    
    # 閥值設定
    threshold = 0.01  # 
    # 上一幀關節位置
    prev_landmark = None
    recent_points = []
    recent_elbow_points = []
    recent_wrist_points = []

    student_data = np.array([])

    plt.pause(0.01)

    idx=0
    rotation_vectors = []
    combined_features=[]
    translation_vectors = []

    maxCnt=300

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 使用DIVX編碼器，也可以選擇其他編碼器，例如


    velocity_data = []
    acceleration_data = []
    previous_wrist_pos= np.array([0,0,0])
    previous_velocity = np.array([0, 0, 0])  # 初始速度為0

    frame_timestamps = []

    filename_with_extension = os.path.basename(video)

    # 分割檔案名和副檔名
    filename, extension = os.path.splitext(video)
    new_filename = filename + "_1" + extension
    new_file_path = os.path.join(os.path.dirname(video), new_filename)

    out = cv2.VideoWriter(new_file_path, fourcc, fps, (width, height))

    while cap.isOpened(): #and idx < maxCnt:
        ret, frame = cap.read()
        if not ret:
            break
        
        OriginalHeight, OriginalWidth = frame.shape[:2]
        new_width = 1200
        ratio = new_width / OriginalWidth
        new_height = int(OriginalHeight * ratio) 
        new_dim = (new_width, new_height)

        frame = cv2.resize(frame, new_dim, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)

        # mediapipe處理image,get關節位置
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # 轉換為秒
        frame_timestamps.append(current_time)

        if results.pose_landmarks:
            # right arm 關節座標
            shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
            right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

            #將正規化座標轉換為像素座標

            current_shoulder = np.array([shoulder.x, shoulder.y, shoulder.z])
            current_wrist = np.array([wrist.x,wrist.y,wrist.z])
            current_elbow = np.array([elbow.x,elbow.y,elbow.z])

            current_Leftshoulder = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])
            current_Leftwrist = np.array([left_wrist.x,left_wrist.y,left_wrist.z])
            current_Leftelbow = np.array([left_elbow.x,left_elbow.y,left_elbow.z])
            
            current_RightHip=np.array([right_hip.x, right_hip.y, right_hip.z])
            current_RightKnee=np.array([right_knee.x, right_knee.y, right_knee.z])
            current_RightAnkle=np.array([right_ankle.x, right_ankle.y, right_ankle.z])

            current_LeftHip=np.array([left_hip.x, left_hip.y, left_hip.z])
            current_LeftKnee=np.array([left_knee.x, left_knee.y, left_knee.z])
            current_LeftAnkle=np.array([left_ankle.x, left_ankle.y, left_ankle.z])

            # 平滑處理
            if smoothed_wrist is None:
                smoothed_wrist = current_wrist
                smoothed_Leftshoulder=current_Leftshoulder
            else:
                smoothed_wrist = alpha * current_wrist + (1 - alpha) * smoothed_wrist
                smoothed_Leftwrist = alpha * current_Leftwrist + (1 - alpha) * smoothed_Leftwrist

            if smoothed_elbow is None:
                smoothed_elbow = current_elbow
                smoothed_Leftelbow=current_Leftelbow
            else:
                smoothed_elbow = alpha * current_elbow + (1 - alpha) * smoothed_elbow
                smoothed_Leftelbow = alpha * current_Leftelbow + (1 - alpha) * smoothed_Leftelbow

            if smoothed_shoulder is None:
                smoothed_shoulder = current_shoulder
                smoothed_Leftwrist=current_Leftwrist
            else:
                smoothed_shoulder = alpha * current_shoulder + (1 - alpha) * smoothed_shoulder
                smoothed_Leftshoulder = alpha * current_Leftshoulder + (1 - alpha) * smoothed_Leftshoulder
                
            
            trajectory_x.append(smoothed_wrist[0])
            trajectory_y.append(smoothed_wrist[1])
            trajectory_z.append(smoothed_wrist[2])

            elbow_wrist_x.append(smoothed_elbow[0])
            elbow_wrist_y.append(smoothed_elbow[1])
            elbow_wrist_z.append(smoothed_elbow[2])

            # Calculate Rotation 
            if len(elbow_coordinates) > 1:
                rotation = quaternion_from_two_vectors(elbow_coordinates[-1], wrist_coordinates[-1])
                rotation_vectors.append(rotation)

            # Calculate Translation
            if len(wrist_coordinates) > 1:
                translation = wrist_coordinates[-1] - wrist_coordinates[-2]
                translation_vectors.append(translation)
            
            shoulder_coordinates.append(smoothed_shoulder)
            elbow_coordinates.append(smoothed_elbow)
            wrist_coordinates.append(smoothed_wrist)

            current_wrist_pos = np.array([wrist.x, wrist.y, wrist.z])
            
            # 計算關節角度
            if idx > 0:  # 確保我們有前一幀的數據來計算向量
                # 計算肩膀到肘部的向量和肘部到手腕的向量
                vector_shoulder_to_elbow = current_elbow - current_shoulder
                vector_elbow_to_wrist = current_wrist - current_elbow

                # 正規化向量
                vector_shoulder_to_elbow_normalized = vector_shoulder_to_elbow / np.linalg.norm(vector_shoulder_to_elbow)
                vector_elbow_to_wrist_normalized = vector_elbow_to_wrist / np.linalg.norm(vector_elbow_to_wrist)

                # 計算左手肩膀到肘部的向量和肘部到手腕的向量
                vector_left_shoulder_to_elbow = current_Leftelbow - current_Leftshoulder
                vector_left_elbow_to_wrist = current_Leftwrist - current_Leftelbow

                # 向量正规化
                vector_left_shoulder_to_elbow_normalized = vector_left_shoulder_to_elbow / np.linalg.norm(vector_left_shoulder_to_elbow)
                vector_left_elbow_to_wrist_normalized = vector_left_elbow_to_wrist / np.linalg.norm(vector_left_elbow_to_wrist)

                # 計算右腳
                vector_right_hip_to_knee = current_RightKnee - current_RightHip
                vector_right_knee_to_ankle = current_RightAnkle - current_RightKnee

                # 向量正规化
                vector_right_hip_to_knee_normalized = vector_right_hip_to_knee / np.linalg.norm(vector_right_hip_to_knee)
                vector_right_knee_to_ankle_normalized = vector_right_knee_to_ankle / np.linalg.norm(vector_right_knee_to_ankle)

                # 計算左腳
                vector_left_hip_to_knee = current_LeftKnee - current_LeftHip
                vector_left_knee_to_ankle = current_LeftAnkle - current_LeftKnee

                # 向量正规化
                vector_left_hip_to_knee_normalized = vector_left_hip_to_knee / np.linalg.norm(vector_left_hip_to_knee)
                vector_left_knee_to_ankle_normalized = vector_left_knee_to_ankle / np.linalg.norm(vector_left_knee_to_ankle)

                
                # 計算兩個向量的點積
                dot_product1 = np.dot(vector_shoulder_to_elbow_normalized, vector_elbow_to_wrist_normalized)
                dot_product2 = np.dot(vector_left_shoulder_to_elbow_normalized, vector_left_elbow_to_wrist_normalized)
                dot_product3 = np.dot(vector_right_hip_to_knee_normalized, vector_right_knee_to_ankle_normalized)
                dot_product4 = np.dot(vector_left_hip_to_knee_normalized, vector_left_knee_to_ankle_normalized)

                # 計算角度
                angle1 = np.arccos(dot_product1)  # 結果是弧度
                angle2 = np.arccos(dot_product2)  # 結果是弧度
                angle3 = np.arccos(dot_product3)  # 結果是弧度
                angle4 = np.arccos(dot_product4)  # 結果是弧度
                angle_degrees = np.degrees(angle1)  # 如果需要，轉換為度
                elbow_angles.append(angle_degrees)

                velocity = current_wrist_pos - previous_wrist_pos  # 以腕部為例
                acceleration = velocity - previous_velocity

                previous_wrist_pos = current_wrist_pos
                previous_velocity = velocity

                velocity_data.append(velocity)
                acceleration_data.append(acceleration)                
                
                combined_features.append(np.hstack((
                    vector_shoulder_to_elbow_normalized, 
                    vector_elbow_to_wrist_normalized,
                    vector_left_shoulder_to_elbow_normalized,
                    vector_left_elbow_to_wrist_normalized,
                    vector_right_hip_to_knee_normalized,
                    vector_right_knee_to_ankle_normalized,
                    vector_left_hip_to_knee_normalized,
                    vector_left_knee_to_ankle_normalized,
                    angle1,angle2,angle3,angle4,
                    velocity,acceleration
                )))

            # 可視化
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_image = frame.copy()
            height, width, _ = annotated_image.shape
            
            #right arm
            drawArmLeg(annotated_image,joint1=current_shoulder,joint2=current_elbow,joint3=current_wrist,color=(0,255,0))

            #left arm
            drawArmLeg(annotated_image,joint1=current_Leftshoulder,joint2=current_Leftelbow,joint3=current_Leftwrist,color=(255,0,0))

            #right leg
            drawArmLeg(annotated_image,joint1=current_RightHip,joint2=current_RightKnee,joint3=current_RightAnkle,color=(0,255,0))
            
            #left leg
            drawArmLeg(annotated_image,joint1=current_LeftHip,joint2=current_LeftKnee,joint3=current_LeftAnkle,color=(255,0,0))

            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Frame', annotated_image)
            out.write(annotated_image)
        else:
            previous_wrist_pos = current_wrist_pos
            previous_velocity = np.array([0, 0, 0])  # 初始速度为0
            cv2.imshow('Frame', frame)
            out.write(annotated_image)

        idx=idx+1
            
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
            
        # if cv2.waitKey(0) == 32:  # 32 is the ASCII code for the space bar
        #     continue
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()            
    # After the loop
    return combined_features,frame_timestamps