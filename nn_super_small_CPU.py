import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import pyautogui
import time

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 初始化模型
input_dim, hidden_dim, output_dim = 42, 128, 4
device = torch.device("cpu")
model = SimpleNN(input_dim, hidden_dim, output_dim).to(device)
model.load_state_dict(
    torch.load("/Users/wangchuhao/Desktop/mlx_based_projects/2_Gesture_Control_System/hand_gesture_nn_model.pth", map_location=device))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

threshold = 0.95
last_click_time = 0
click_cooldown = 1.0

def predict_gesture(landmarks):
    input_data = torch.tensor([landmarks], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_data)
        confidence, predicted_class = torch.max(torch.softmax(output, dim=1), dim=1)
    return predicted_class.item(), confidence.item()

def handle_gesture(gesture, hand_landmarks):
    global last_click_time
    current_time = time.time()

    if gesture == 0 and current_time - last_click_time > click_cooldown:
        pyautogui.click()
        last_click_time = current_time
    elif gesture == 1:
        x, y = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
        screenWidth, screenHeight = pyautogui.size()
        pyautogui.moveTo(screenWidth - int(x * screenWidth), int(y * screenHeight))
    elif gesture == 2:
        thumb_tip, index_tip = hand_landmarks.landmark[4], hand_landmarks.landmark[8]
        distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
        if distance > 0.1:
            pyautogui.hotkey('ctrl', '+')
        elif distance < 0.05:
            pyautogui.hotkey('ctrl', '-')
    elif gesture == 3:
        middle_tip = hand_landmarks.landmark[12].y
        if middle_tip > 0.6:
            pyautogui.scroll(-150)
        elif middle_tip < 0.4:
            pyautogui.scroll(150)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        landmarks = [l for hand_landmarks in results.multi_hand_landmarks for l in hand_landmarks.landmark[:21] for l in (l.x, l.y)]
        gesture, confidence = predict_gesture(landmarks[:42])

        if confidence >= threshold:
            handle_gesture(gesture, results.multi_hand_landmarks[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()