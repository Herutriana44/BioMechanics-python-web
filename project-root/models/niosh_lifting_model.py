import numpy as np

def calculate_niosh_lifting_equation(pose_landmarks, frame_shape):
    # Ambil landmark yang diperlukan, contoh: bahu dan siku
    left_shoulder = pose_landmarks.landmark[11]
    left_elbow = pose_landmarks.landmark[13]

    # Menghitung jarak horizontal dan vertikal (dalam piksel)
    H = abs(left_shoulder.x - left_elbow.x) * frame_shape[1]
    V = abs(left_shoulder.y - left_elbow.y) * frame_shape[0]

    # Load Constant (LC)
    LC = 23  # kg
    HM = 10 / H  # Horizontal Multiplier
    VM = 1 - 0.003 * abs(V - 75)  # Vertical Multiplier
    DM = 0.82 + 4.5 / (2.2 * H)  # Distance Multiplier

    RWL = LC * HM * VM * DM
    actual_weight = 25  # kg, contoh
    LI = actual_weight / RWL

    return {"RWL": RWL, "LI": LI}
