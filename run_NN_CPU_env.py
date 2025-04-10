'''
This is for NN-based evaluation
'''
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import pyautogui
import time
import pygame

# 初始化 pygame  
pygame.init()  
clock = pygame.time.Clock()

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

input_dim, hidden_dim, output_dim = 42, 128, 4
device = torch.device("cpu")
model = SimpleNN(input_dim, hidden_dim, output_dim).to(device)
model.load_state_dict(
    torch.load("/Users/wangchuhao/Desktop/mlx_based_projects/2_Gesture_Control_System/hand_gesture_nn_model.pth", map_location=device))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
threshold = 0.98
five_threshold = 0.99
initial_y = 0
initial_distance = 0
movement = False
last_click_time = 0
click_cooldown = 3.0


def predict_gesture(landmarks):
    input_data = torch.tensor([landmarks], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_data)
        softmax = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(softmax, dim=1)
    return predicted_class.item(), confidence.item()

def is_valid_five_gesture(hand_landmarks):
    thumb_open =True
    return thumb_open


def handle_gesture(gesture, hand_landmarks):
    global movement, initial_y, initial_distance, last_click_time
    current_time = time.time()

    if gesture == 0 and current_time - last_click_time > click_cooldown:
        if is_valid_five_gesture(hand_landmarks):
            pyautogui.click()
            last_click_time = current_time
    elif gesture == 1:
        x, y = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
        screenWidth, screenHeight = pyautogui.size()
        pyautogui.moveTo(screenWidth - int(x * screenWidth), int(y * screenHeight))
    elif gesture == 2:
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        current_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
        if not movement:
            initial_distance, movement = current_distance, True
        else:
            if current_distance - initial_distance > 0.01:
                pyautogui.hotkey('command', '+')
            elif current_distance - initial_distance < -0.01:
                pyautogui.hotkey('command', '-')
            initial_distance = current_distance
    elif gesture == 3:  # 滚动
        middle_tip = hand_landmarks.landmark[12].y
        if not movement:
            initial_y, movement = middle_tip, True
        else:
            if middle_tip - initial_y > 0.03:
                pyautogui.scroll(25)
            elif middle_tip - initial_y < -0.03:
                pyautogui.scroll(-25)
            initial_y = middle_tip

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = [l for hand_landmarks in results.multi_hand_landmarks for l in hand_landmarks.landmark[:21] for l in
                     (l.x, l.y)]
        gesture, confidence = predict_gesture(landmarks[:42])

        if (gesture == 0 and confidence >= five_threshold) or (gesture != 0 and confidence >= threshold):
            handle_gesture(gesture, results.multi_hand_landmarks[0])
    else:
        movement = False

    clock.tick(15)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()