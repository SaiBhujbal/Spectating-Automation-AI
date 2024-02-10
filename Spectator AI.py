import cv2
import numpy as np
import pyautogui
import time
from mss import mss
from ultralytics import YOLO
import random
import socket

# Establish a continuous connection
class ContinuousClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client_socket = socket.socket()
        self.connect()

    def connect(self):
        try:
            self.client_socket.connect((self.host, self.port))
            print("Connected to server.")
        except Exception as e:
            print(f"Connection failed: {e}")

    def send_key_press(self, key):
        try:
            self.client_socket.send(key.encode())
            response = self.client_socket.recv(1024).decode()
            print('Received from server:', response)
        except Exception as e:
            print(f"Failed to send key press: {e}")
            self.connect()  # Attempt to reconnect if sending fails

    def close(self):
        self.client_socket.close()


# Initialize the YOLO model
model = YOLO('model_- 5 february 2024 12_04.pt')

# Define constants for frame rate, critical distance, and switch intervals
frame_rate = 20
critical_distance = 75
switch_interval = 3
random_switch_interval = 8
alternate_limit = 2
delay_after_less_contours = 4  # Delay of 4 seconds when less than 3 contours

# Initialize global variables for tracking state
current_pair = None
last_switch_time = 0
recent_switched_ids = []
last_random_switch_time = time.time()

last_switched_ids = []  # Track the IDs switched to in the last random switch
last_chosen_pair = None  # Track the last chosen pair for switching
alternate_switch = False  # Flag to alternate switch within the same pair
switch_count = 0  # Count alternations for the current pair
last_switch_time = 0
alternate_switch = False

def select_screen_area():
    with mss() as sct:
        monitor = sct.monitors[1]
        screen = np.array(sct.grab(monitor))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
        roi = cv2.selectROI("Screen Capture", screen, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        return {"top": int(roi[1]), "left": int(roi[0]), "width": int(roi[2]), "height": int(roi[3])}

def capture_screen(monitor_area):
    with mss() as sct:
        return cv2.cvtColor(np.array(sct.grab(monitor_area)), cv2.COLOR_BGRA2RGB)

def detect_objects(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(frame_rgb)
    centroids, team1, team2, class_ids, player_bboxes = [], [], [], [], []
    for det in results[0].boxes:
        bbox = det.xyxy[0].cpu().numpy()
        player_bboxes.append(bbox)
        x1, y1, x2, y2 = bbox
        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        centroids.append(centroid)
        class_id = str(int(det.cls))
        if class_id in ["1", "2", "3", "4", "5"]:
            team1.append((centroid, class_id))
        elif class_id in ["6", "7", "8", "9", "0"]:
            team2.append((centroid, class_id))
        class_ids.append(class_id)
    return centroids, team1, team2, class_ids, player_bboxes

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def visualize_detections(frame, centroids, class_ids, filtered_contours, chosen_pair=None, min_distance=None):
    cv2.drawContours(frame, filtered_contours, -1, (0, 255, 0), 1)
    for centroid, class_id in zip(centroids, class_ids):
        centroid_int = (int(centroid[0]), int(centroid[1]))
        cv2.circle(frame, centroid_int, 5, (255, 0, 0), -1)
        cv2.putText(frame, str(class_id), (centroid_int[0] + 10, centroid_int[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    if chosen_pair and min_distance is not None:
        p1_centroid, p2_centroid = chosen_pair[0][0], chosen_pair[1][0]
        cv2.line(frame, (int(p1_centroid[0]), int(p1_centroid[1])), (int(p2_centroid[0]), int(p2_centroid[1])), (0, 255, 0), 2)
        mid_point = ((int(p1_centroid[0]) + int(p2_centroid[0])) // 2, (int(p1_centroid[1]) + int(p2_centroid[1])) // 2)
        cv2.putText(frame, f"{min_distance:.2f}px", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Detections", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def detect_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_out_player_dots(contours, player_bboxes):
    filtered_contours = []
    for contour in contours:
        keep_contour = True
        for bbox in player_bboxes:
            distance = cv2.pointPolygonTest(contour, ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2), True)
            if distance >= 0:
                keep_contour = False
                break
        if keep_contour:
            filtered_contours.append(contour)
    return filtered_contours

def bbox_overlap(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1b, y1b, x2b, y2b = bbox2
    return not (x2 < x1b or x2b < x1 or y2 < y1b or y2b < y1)


# def send_key_press(key):
#     host = '192.168.127.113'  # as both code is running on same pc
#     port = 5000  # socket server port number

#     client_socket = socket.socket()  # instantiate
#     client_socket.connect((host, port))  # connect to the server

#     client_socket.send(key.encode())  # send message
#     data = client_socket.recv(1024).decode()  # receive response

#     print('Received from server: ' + data)  # show in terminal

#     client_socket.close()  # close the connection
def find_clusters(players, threshold=30):
    clusters = []
    for i, player1 in enumerate(players):
        for j, player2 in enumerate(players[i + 1:], start=i + 1):
            if calculate_distance(player1[0], player2[0]) < threshold:
                clusters.append((player1, player2))
    return clusters

def is_within_same_team(player1, player2, team1_ids, team2_ids):
    return (player1[1] in team1_ids and player2[1] in team1_ids) or (player1[1] in team2_ids and player2[1] in team2_ids)

def make_decision(frame, centroids, team1, team2, class_ids, player_bboxes, filtered_contours, client):
    global current_pair, last_switch_time, last_random_switch_time, recent_switched_ids
    global switch_count, last_chosen_pair, alternate_switch
    current_time = time.time()
    min_distance = float('inf')
    chosen_pair = None
    should_random_switch = True

    # Identifiers for team members to assist in checking team membership during clustering
    team1_ids = [player[1] for player in team1]
    team2_ids = [player[1] for player in team2]

    # Find clusters within each team
    team1_clusters = find_clusters(team1, 20)  # Clustering threshold within same team
    team2_clusters = find_clusters(team2, 20)
    clustering_scenario = team1_clusters or team2_clusters

    if clustering_scenario and current_time - last_switch_time > 1:  # 1-second delay for cluster switching
        all_players = team1 + team2
        # Filter out players that have been switched to recently
        available_players = [player for player in all_players if player[1] not in recent_switched_ids]
        if not available_players:  # If no available players, skip this cycle
            return

        # Calculate distances between all available players, prioritizing those not recently switched to
        distances = []
        for i, player1 in enumerate(available_players):
            for player2 in (team1 + team2)[i + 1:]:
                if player1[1] != player2[1] and not is_within_same_team(player1, player2, team1_ids, team2_ids):
                    dist = calculate_distance(player1[0], player2[0])
                    distances.append(((player1, player2), dist))

        if distances:
            # Choose the pair with the shortest distance
            chosen_pair, _ = min(distances, key=lambda x: x[1])
            chosen_id = chosen_pair[0][1]
            client.send_key_press(chosen_id)
            print(f"Cluster switch to: {chosen_id}")
            recent_switched_ids.append(chosen_id)
            if len(recent_switched_ids) > 10:
                recent_switched_ids.pop(0)
            last_switch_time = current_time
            return
    # Check for direct visibility with no obstacles for pairs beyond the critical distance
    for i, p1 in enumerate(team1):
        for j, p2 in enumerate(team2):
            dist = calculate_distance(p1[0], p2[0])
            if dist > critical_distance:
                # Simplified direct visibility check using contours
                if not any(cv2.pointPolygonTest(contour, ((p1[0][0]+p2[0][0])/2, (p1[0][1]+p2[0][1])/2), False) > 0 for contour in filtered_contours):
                    if dist < min_distance:
                        min_distance = dist
                        chosen_pair = (p1, p2)
                        should_random_switch = False

    # If a pair with direct visibility and beyond critical distance is found
    if chosen_pair and not should_random_switch and current_time - last_switch_time > delay_after_less_contours:
        chosen_id = chosen_pair[0][1]
        client.send_key_press(chosen_id)
        print(f"Switched to: {chosen_id}, distance: {min_distance:.2f}px, due to direct visibility without obstacles.")
        last_switch_time = current_time
        current_pair = chosen_pair
        switch_count = 0
        return  # Return early to enforce the delay before considering other conditions

    # Continue with the original logic if no direct visibility pair was found or delay not met
    if current_time - last_switch_time > switch_interval:
        for p1 in team1:
            for p2 in team2:
                dist = calculate_distance(p1[0], p2[0])
                if dist < min_distance:
                    min_distance = dist
                    chosen_pair = (p1, p2)
                    should_random_switch = min_distance > critical_distance

        if chosen_pair:
            if chosen_pair != current_pair:
                current_pair = chosen_pair
                last_chosen_pair = chosen_pair
                switch_count = 0
                alternate_switch = False

            if min_distance <= critical_distance:
                chosen_id = chosen_pair[0][1]
                client.send_key_press(chosen_id)
                print(f"Switched to: {chosen_id}, distance: {min_distance:.2f}px")
                last_switch_time = current_time
                should_random_switch = False

    # Implement random switching if no pair chosen by previous criteria
    if should_random_switch and current_time - last_random_switch_time > random_switch_interval:
        available_ids = [cid for cid in class_ids if cid not in recent_switched_ids]
        if available_ids:
            random_id = random.choice(available_ids)
            client.send_key_press(random_id)
            print(f"Randomly switched to: {random_id}")
            last_random_switch_time = current_time
            recent_switched_ids.append(random_id)
            if len(recent_switched_ids) > 10:
                recent_switched_ids.pop(0)
            current_pair = None
            switch_count = 0

    # Visualization of detections
    visualize_detections(frame, centroids, class_ids, filtered_contours, chosen_pair, min_distance)

def main():
    # Define server details
    server_ip = "192.168.127.113"  # Change this to your server's IP address
    server_port = 5000

    # Initialize the continuous client
    client = ContinuousClient(server_ip, server_port)

    monitor_area = select_screen_area()
    if monitor_area["width"] == 0 or monitor_area["height"] == 0:
        print("Screen area selection failed. Exiting...")
        client.close()  # Close the connection when exiting
        return

    print("Starting detection and auto-switching...")
    initial_frame = capture_screen(monitor_area)
    map_contours = detect_edges(initial_frame)
    _, team1, team2, class_ids, player_bboxes = detect_objects(initial_frame)
    filtered_contours = filter_out_player_dots(map_contours, player_bboxes)

    while True:
        frame = capture_screen(monitor_area)
        centroids, team1, team2, class_ids, player_bboxes = detect_objects(frame)
        contours = detect_edges(frame)  # Detect contours for each frame if dynamic; else use filtered_contours
        make_decision(frame, centroids, team1, team2, class_ids, player_bboxes, filtered_contours, client)  # Pass the client here
        time.sleep(1 / frame_rate)

    # If you have a way to exit the loop, remember to close the client connection
    # client.close()

if __name__ == "__main__":
    main()

