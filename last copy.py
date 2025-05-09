import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import os
import pygame.sndarray
from collections import deque
from settings_page import SettingsPage
import tkinter as tk
import json
import threading
import sys

# Initialize pygame
pygame.init()
pygame.mixer.init()

# Constants
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 50, 255)
DARK_GRAY = (30, 30, 30)
LIGHT_GRAY = (200, 200, 200)
TEAL = (0, 128, 128)
DARK_TEAL = (0, 77, 77)
YELLOW = (255, 255, 0)

# UI Elements
PANEL_WIDTH = 300
HEADER_HEIGHT = 60
FOOTER_HEIGHT = 40

# Eye detection constants
EAR_THRESHOLD = 0.15
DROWSY_FRAMES_THRESHOLD = 30
BLINK_THRESHOLD = 0.2
ALERT_COOLDOWN = 3.0

# Add LoadingScreen class from drowsinessui.py
class LoadingScreen:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Uykusuzluk Takip Sistemi - Yükleniyor")
        
        # Fonts
        self.title_font = pygame.font.SysFont("Arial", 48, bold=True)
        self.normal_font = pygame.font.SysFont("Arial", 24)
        
        # Loading animation variables
        self.progress = 0
        self.loading_text = "Sistem yükleniyor..."
        self.tasks = [
            "MediaPipe yüz tanıma modülü başlatılıyor...",
            "Kamera ayarları yapılandırılıyor...",
            "Kullanıcı arayüzü hazırlanıyor...",
            "Uykusuzluk algılama algoritması başlatılıyor...",
            "Uyarı sistemi hazırlanıyor..."
        ]
        self.current_task = 0
        self.clock = pygame.time.Clock()
        self.loading_done = False
        
    def draw(self):
        # Fill background
        self.screen.fill(DARK_GRAY)
        
        # Draw title
        title = self.title_font.render("Uykusuzluk Takip Sistemi", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3))
        self.screen.blit(title, title_rect)
        
        # Draw progress bar background
        bar_width = SCREEN_WIDTH // 2
        bar_height = 20
        bar_x = (SCREEN_WIDTH - bar_width) // 2
        bar_y = SCREEN_HEIGHT // 2
        pygame.draw.rect(self.screen, BLACK, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 2)
        
        # Draw progress bar fill
        fill_width = int(bar_width * self.progress)  # Corrected proportional calculation
        pygame.draw.rect(self.screen, TEAL, (bar_x, bar_y, fill_width, bar_height))
        
        # Draw loading text
        if self.current_task < len(self.tasks):
            task_text = self.tasks[self.current_task]
        else:
            task_text = "Sistem başlatılıyor..."
            
        loading_label = self.normal_font.render(task_text, True, LIGHT_GRAY)
        loading_rect = loading_label.get_rect(center=(SCREEN_WIDTH // 2, bar_y + 50))
        self.screen.blit(loading_label, loading_rect)
        
        # Draw progress percentage
        percent_text = self.normal_font.render(f"{int(self.progress * 100)}%", True, WHITE)
        percent_rect = percent_text.get_rect(center=(SCREEN_WIDTH // 2, bar_y - 30))
        self.screen.blit(percent_text, percent_rect)
        
        # Draw animated dots
        dots = "." * ((pygame.time.get_ticks() // 500) % 4)
        dots_text = self.normal_font.render(dots, True, WHITE)
        dots_rect = dots_text.get_rect(midleft=(loading_rect.right + 5, loading_rect.centery))
        self.screen.blit(dots_text, dots_rect)
        
        pygame.display.flip()
    
    def update(self, progress):
        self.progress = progress
        self.current_task = min(int(progress * len(self.tasks)), len(self.tasks) - 1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.draw()
        self.clock.tick(FPS)

# Add initialize_mediapipe and load_alert_sound functions from drowsinessui.py
def initialize_mediapipe():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    return mp_face_mesh, face_mesh, mp_drawing, drawing_spec

def load_alert_sound():
    try:
        alert_sound = pygame.mixer.Sound("alert.wav")
    except:
        print("Alert sound file not found. Creating a default beep sound.")
        # Create a simple beep sound if the file is not found
        pygame.mixer.init(frequency=44100, size=-16, channels=1)
        sample_rate = 44100
        duration = 0.5  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * 440 * t)
        tone = (tone * 32767).astype(np.int16)
        tone_bytes = tone.tobytes()
        alert_sound = pygame.mixer.Sound(buffer=tone_bytes)
    return alert_sound

class Button:
    def __init__(self, x, y, width, height, text, color, hover_color, text_color=WHITE, font_size=20):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = pygame.font.SysFont("Arial", font_size)
        self.is_hovered = False
        
    def draw(self, screen):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, WHITE, self.rect, 2, border_radius=5)
        
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        
    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(pos)
        return False

# Modify ModernDrowsinessDetection class to include loading screen
class ModernDrowsinessDetection:
    def __init__(self):
        # Add loading screen initialization
        self.loading_screen = LoadingScreen()
        self.loading_screen.update(0.0)
        
        # Eye landmark indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Initialize components with progress updates
        self.loading_screen.update(0.1)
        global EAR_THRESHOLD, DROWSY_FRAMES_THRESHOLD, BLINK_THRESHOLD
        # Eye detection constants
        EAR_THRESHOLD = 0.15
        DROWSY_FRAMES_THRESHOLD = 30
        BLINK_THRESHOLD = 0.2

        # Load settings from JSON file
        try:
            with open("saved_settings.json", "r") as f:
                self.settings = json.load(f)
        except FileNotFoundError:
            self.settings = {}

        # Ensure all required settings have default values
        self.settings.setdefault("EAR_THRESHOLD", EAR_THRESHOLD)
        self.settings.setdefault("DROWSY_FRAMES_THRESHOLD", DROWSY_FRAMES_THRESHOLD)
        self.settings.setdefault("BLINK_THRESHOLD", BLINK_THRESHOLD)
        self.settings.setdefault("YAWN_THRESHOLD", 30)  # Default yawn threshold
        self.settings.setdefault("YAWN_COOLDOWN", 5)    # Default yawn cooldown
        self.settings.setdefault("EYE_CLOSURE_DURATION", 3.0)  # Default eye closure duration
        
        self.loading_screen.update(0.2)
        
        # Initialize MediaPipe (potentially slow operation)
        self.mp_face_mesh, self.face_mesh, self.mp_drawing, self.drawing_spec = initialize_mediapipe()
        self.loading_screen.update(0.4)
        
        # Load alert sound
        self.alert_sound = load_alert_sound()
        self.loading_screen.update(0.5)
        
        # Setup webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            pygame.quit()
            sys.exit()
            
        # Get webcam dimensions
        self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.loading_screen.update(0.6)
        
        # Initialize display
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Uykusuzluk Takip Sistemi")
        
        # Calculate video display dimensions
        self.video_width = SCREEN_WIDTH - PANEL_WIDTH
        self.video_height = SCREEN_HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT
        self.aspect_ratio = self.cam_width / self.cam_height
        self.display_height = self.video_height
        self.display_width = int(self.display_height * self.aspect_ratio)
        
        if self.display_width > self.video_width:
            self.display_width = self.video_width
            self.display_height = int(self.display_width / self.aspect_ratio)
        
        # Calculate video position to center it
        self.video_x = (self.video_width - self.display_width) // 2
        self.video_y = HEADER_HEIGHT + (self.video_height - self.display_height) // 2
        
        self.loading_screen.update(0.7)
        
        # Fonts
        self.title_font = pygame.font.SysFont("Arial", 24, bold=True)
        self.header_font = pygame.font.SysFont("Arial", 20, bold=True)
        self.normal_font = pygame.font.SysFont("Arial", 16)
        self.small_font = pygame.font.SysFont("Arial", 14)
        
        # Buttons
        button_y = SCREEN_HEIGHT - FOOTER_HEIGHT - 50
        self.settings_button = Button(SCREEN_WIDTH - PANEL_WIDTH + 20, button_y, 120, 35, "Ayarlar", DARK_TEAL, TEAL)
        self.quit_button = Button(SCREEN_WIDTH - PANEL_WIDTH + 160, button_y, 120, 35, "Çıkış", DARK_TEAL, TEAL)
        
        self.loading_screen.update(0.8)
        
        # Drowsiness detection variables
        self.drowsy_frames = 0
        self.alert_active = False
        self.last_alert_time = 0
        self.blink_start = False
        self.blink_frames = 0
        self.total_blinks = 0
        self.blink_times = deque(maxlen=60)
        self.start_time = time.time()
        self.ear_values = deque(maxlen=100)  # For plotting EAR graph
        
        # Session stats
        self.session_start_time = time.time()
        self.drowsy_alerts = 0
        self.max_blink_rate = 0
        self.current_ear = 0
        
        # App state
        self.running = True
        self.clock = pygame.time.Clock()
        self.face_detected = False
        
        # Show a welcome animation
        self.loading_screen.update(0.9)
        self.show_welcome_animation()
        self.loading_screen.update(1.0)
        
    def show_welcome_animation(self):
        """Show a welcome animation before starting the main application"""
        # Clean the screen
        self.screen.fill(DARK_GRAY)
        
        # Prepare welcome text
        welcome_font = pygame.font.SysFont("Arial", 60, bold=True)
        welcome_text = welcome_font.render("Hoş Geldiniz!", True, WHITE)
        subtitle_font = pygame.font.SysFont("Arial", 30)
        subtitle_text = subtitle_font.render("Uykusuzluk Takip ve Uyarı Sistemi", True, TEAL)
        
        # Animation parameters
        start_time = time.time()
        animation_duration = 2.0  # seconds
        
        while time.time() - start_time < animation_duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    return  # Skip animation on key press
            
            # Calculate animation progress (0.0 to 1.0)
            progress = min((time.time() - start_time) / animation_duration, 1.0)
            
            # Clear screen
            self.screen.fill(DARK_GRAY)
            
            # Animate welcome text
            welcome_alpha = int(255 * progress)
            welcome_text.set_alpha(welcome_alpha)
            
            # Position text
            w_rect = welcome_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 40))
            s_rect = subtitle_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30))
            
            # Draw text
            self.screen.blit(welcome_text, w_rect)
            
            # Show subtitle only in second half of animation
            if progress > 0.5:
                subtitle_alpha = int(255 * (progress - 0.5) * 2)
                subtitle_text.set_alpha(subtitle_alpha)
                self.screen.blit(subtitle_text, s_rect)
            
            pygame.display.flip()
            self.clock.tick(60)

    def calculate_ear(self, eye_landmarks):
        p1 = eye_landmarks[0]
        p2 = eye_landmarks[1]
        p3 = eye_landmarks[2]
        p4 = eye_landmarks[3]
        p5 = eye_landmarks[4]
        p6 = eye_landmarks[5]
        
        distance_p2p6 = np.linalg.norm(np.array(p2) - np.array(p6))
        distance_p3p5 = np.linalg.norm(np.array(p3) - np.array(p5))
        distance_p1p4 = np.linalg.norm(np.array(p1) - np.array(p4))
        
        ear = (distance_p2p6 + distance_p3p5) / (2.0 * distance_p1p4)
        return ear
    
    def calculate_blinks_per_minute(self):
        current_time = time.time()
        while self.blink_times and current_time - self.blink_times[0] > 60:
            self.blink_times.popleft()
        
        return len(self.blink_times)
    
    def process_frame(self, frame):
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        ear = 0
        self.face_detected = False
        
        if results.multi_face_landmarks:
            self.face_detected = True
            for face_landmarks in results.multi_face_landmarks:
                h, w, c = frame.shape
                
                left_eye_coords = []
                right_eye_coords = []
                
                for idx in self.LEFT_EYE:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    left_eye_coords.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                for idx in self.RIGHT_EYE:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    right_eye_coords.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                left_ear = self.calculate_ear(left_eye_coords)
                right_ear = self.calculate_ear(right_eye_coords)
                
                ear = (left_ear + right_ear) / 2.0
                self.current_ear = ear
                self.ear_values.append(ear)
                
                if ear < EAR_THRESHOLD:
                    self.drowsy_frames += 1
                    
                    if self.drowsy_frames >= DROWSY_FRAMES_THRESHOLD:
                        cv2.putText(frame, "UYKUSUZLUK UYARISI!", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        current_time = time.time()
                        if not self.alert_active or (current_time - self.last_alert_time) > ALERT_COOLDOWN:
                            self.alert_sound.play()
                            self.alert_active = True
                            self.last_alert_time = current_time
                            self.drowsy_alerts += 1
                else:
                    self.drowsy_frames = 0
                    self.alert_active = False
                
                if ear < BLINK_THRESHOLD and not self.blink_start:
                    self.blink_start = True
                elif ear < BLINK_THRESHOLD and self.blink_start:
                    self.blink_frames += 1
                elif ear >= BLINK_THRESHOLD and self.blink_start:
                    if self.blink_frames > 1:
                        self.total_blinks += 1
                        self.blink_times.append(time.time())
                    self.blink_start = False
                    self.blink_frames = 0
        
        # Convert BGR to RGB for Pygame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, ear
    
    def draw_ear_graph(self, x, y, width, height):
        if not self.ear_values:
            return
            
        # Draw graph background
        pygame.draw.rect(self.screen, BLACK, (x, y, width, height))
        pygame.draw.rect(self.screen, LIGHT_GRAY, (x, y, width, height), 1)
        
        # Draw threshold line
        threshold_y = y + height - int(height * EAR_THRESHOLD / 0.4)
        pygame.draw.line(self.screen, RED, (x, threshold_y), (x + width, threshold_y), 1)
        
        # Draw graph
        points = []
        for i, ear in enumerate(self.ear_values):
            point_x = x + i * (width / len(self.ear_values))
            point_y = y + height - int(height * ear / 0.4)  # Normalize EAR to graph height
            points.append((point_x, point_y))
            
        if len(points) > 1:
            pygame.draw.lines(self.screen, GREEN, False, points, 2)
        
        # Draw labels
        label = self.small_font.render("EAR Threshold", True, RED)
        self.screen.blit(label, (x + 5, threshold_y - 15))
        
    def draw_stats_panel(self):
        # Draw panel background
        panel_rect = pygame.Rect(SCREEN_WIDTH - PANEL_WIDTH, 0, PANEL_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, DARK_GRAY, panel_rect)
        
        # Title
        title = self.title_font.render("Uykusuzluk Takip Sistemi", True, WHITE)
        self.screen.blit(title, (SCREEN_WIDTH - PANEL_WIDTH + 20, 20))
        
        # Session info
        session_time = time.time() - self.session_start_time
        hours, remainder = divmod(int(session_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        
        y_pos = 70
        session_label = self.header_font.render("Oturum İstatistikleri", True, WHITE)
        self.screen.blit(session_label, (SCREEN_WIDTH - PANEL_WIDTH + 20, y_pos))
        
        y_pos += 30
        time_label = self.normal_font.render(f"Geçen Zaman: {time_str}", True, LIGHT_GRAY)
        self.screen.blit(time_label, (SCREEN_WIDTH - PANEL_WIDTH + 20, y_pos))
        
        y_pos += 25
        blinks_per_min = self.calculate_blinks_per_minute()
        if blinks_per_min > self.max_blink_rate:
            self.max_blink_rate = blinks_per_min
            
        blink_label = self.normal_font.render(f"Dakika Başına Göz Kırpma: {blinks_per_min}", True, LIGHT_GRAY)
        self.screen.blit(blink_label, (SCREEN_WIDTH - PANEL_WIDTH + 20, y_pos))
        
        y_pos += 25
        total_blink_label = self.normal_font.render(f"Toplam Göz Kırpma: {self.total_blinks}", True, LIGHT_GRAY)
        self.screen.blit(total_blink_label, (SCREEN_WIDTH - PANEL_WIDTH + 20, y_pos))
        
        y_pos += 25
        max_blink_label = self.normal_font.render(f"Toplam Göz Kırpma: {self.max_blink_rate}/min", True, LIGHT_GRAY)
        self.screen.blit(max_blink_label, (SCREEN_WIDTH - PANEL_WIDTH + 20, y_pos))
        
        y_pos += 25
        alerts_label = self.normal_font.render(f"Uykusuzluk Uyarıları: {self.drowsy_alerts}", True, LIGHT_GRAY)
        self.screen.blit(alerts_label, (SCREEN_WIDTH - PANEL_WIDTH + 20, y_pos))
        
        # Current EAR value
        y_pos += 40
        ear_header = self.header_font.render("Current EAR", True, WHITE)
        self.screen.blit(ear_header, (SCREEN_WIDTH - PANEL_WIDTH + 20, y_pos))
        
        y_pos += 30
        ear_value = self.normal_font.render(f"{self.current_ear:.3f}", True, GREEN if self.current_ear > EAR_THRESHOLD else RED)
        self.screen.blit(ear_value, (SCREEN_WIDTH - PANEL_WIDTH + 20, y_pos))
        
        # EAR Graph
        y_pos += 40
        graph_header = self.header_font.render("EAR History", True, WHITE)
        self.screen.blit(graph_header, (SCREEN_WIDTH - PANEL_WIDTH + 20, y_pos))
        
        y_pos += 30
        self.draw_ear_graph(SCREEN_WIDTH - PANEL_WIDTH + 20, y_pos, PANEL_WIDTH - 40, 120)
        
        # Draw status indicator
        y_pos += 150
        status_header = self.header_font.render("Durum", True, WHITE)
        self.screen.blit(status_header, (SCREEN_WIDTH - PANEL_WIDTH + 20, y_pos))
        
        y_pos += 30
        if self.drowsy_frames >= DROWSY_FRAMES_THRESHOLD:
            status_text = "UYKUSUZLUK FARKEDİLDİ "
            status_color = RED
        elif not self.face_detected:
            status_text = "YÜZ TANIMLANAMADI"
            status_color = YELLOW
        else:
            status_text = "UYARI SİSTEMİ AKTİF"
            status_color = GREEN
            
        status_label = self.normal_font.render(status_text, True, status_color)
        self.screen.blit(status_label, (SCREEN_WIDTH - PANEL_WIDTH + 20, y_pos))
    
        # Draw buttons
        self.settings_button.draw(self.screen)
        self.quit_button.draw(self.screen)
    
    def draw_header(self):
        # Draw header background
        header_rect = pygame.Rect(0, 0, SCREEN_WIDTH - PANEL_WIDTH, HEADER_HEIGHT)
        pygame.draw.rect(self.screen, TEAL, header_rect)
        
        # Draw title
        title = self.title_font.render("Sürücü Uykusuzluk Takip Ve Uyarı Sistemi", True, WHITE)
        self.screen.blit(title, (20, (HEADER_HEIGHT - title.get_height()) // 2))
    
    def draw_footer(self):
        # Draw footer background
        footer_rect = pygame.Rect(0, SCREEN_HEIGHT - FOOTER_HEIGHT, SCREEN_WIDTH - PANEL_WIDTH, FOOTER_HEIGHT)
        pygame.draw.rect(self.screen, DARK_TEAL, footer_rect)
        
        # Draw status text
        status_text = "Çıkmak İçin 'Q' Basın. | İstatistikleri Sıfırlamak İçin Space Basın."
        status = self.small_font.render(status_text, True, WHITE)
        self.screen.blit(status, (20, SCREEN_HEIGHT - FOOTER_HEIGHT + (FOOTER_HEIGHT - status.get_height()) // 2))
    
    def open_settings(self):
        def save_settings(new_settings):
            self.settings.update(new_settings)
            global EAR_THRESHOLD, DROWSY_FRAMES_THRESHOLD, BLINK_THRESHOLD
            EAR_THRESHOLD = self.settings["EAR_THRESHOLD"]
            DROWSY_FRAMES_THRESHOLD = self.settings["DROWSY_FRAMES_THRESHOLD"]
            BLINK_THRESHOLD = self.settings["BLINK_THRESHOLD"]

        root = tk.Tk()
        SettingsPage(root, self.settings, save_settings)
        root.mainloop()

    def run(self):
        while self.running:
            # Process events
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        # Reset stats
                        self.session_start_time = time.time()
                        self.drowsy_alerts = 0
                        self.total_blinks = 0
                        self.max_blink_rate = 0
                        self.blink_times.clear()
                
                # Check button clicks
                self.settings_button.check_hover(mouse_pos)
                self.quit_button.check_hover(mouse_pos)
                
                if self.quit_button.is_clicked(mouse_pos, event):
                    self.running = False
                if self.settings_button.is_clicked(mouse_pos, event):
                    self.open_settings()
            
            # Capture webcam frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Flip horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame for drowsiness detection
            processed_frame, ear = self.process_frame(frame)
            
            # Resize processed_frame to match frame_surface dimensions
            processed_frame = cv2.resize(processed_frame, (self.cam_width, self.cam_height))
            
            # Ensure processed_frame is in the correct format for Pygame
            processed_frame = np.transpose(processed_frame, (1, 0, 2))  # Ensure correct channel order
            processed_frame = processed_frame.astype(np.uint8)  # Ensure correct data type
            
            # Convert to Pygame surface
            frame_surface = pygame.Surface((self.cam_width, self.cam_height))
            pygame.surfarray.blit_array(frame_surface, processed_frame)
            
            # Scale to display size
            display_surface = pygame.transform.scale(frame_surface, (self.display_width, self.display_height))
            
            # Clear screen
            self.screen.fill(BLACK)
            
            # Draw header
            self.draw_header()
            
            # Draw footer
            self.draw_footer()
            
            # Draw webcam feed
            self.screen.blit(display_surface, (self.video_x, self.video_y))
            
            # Draw stats panel
            self.draw_stats_panel()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
        
        # Clean up
        self.cap.release()
        pygame.quit()

if __name__ == "__main__":
    app = ModernDrowsinessDetection()
    app.run()