import tkinter as tk
from tkinter import ttk
import json

class SettingsPage:
    def __init__(self, root, initial_settings, on_save):
        self.root = root
        self.root.title("Settings")
        self.root.geometry("400x300")
        
        self.settings = initial_settings
        self.on_save = on_save

        # EAR Threshold
        tk.Label(root, text="EAR Threshold").pack(anchor="w", padx=20, pady=(10, 0))
        self.ear_slider = ttk.Scale(root, from_=0.05, to=0.25, value=self.settings["EAR_THRESHOLD"], orient="horizontal")
        self.ear_slider.pack(fill="x", padx=20)
        tk.Label(root, text="0.05").place(x=20, y=50)
        tk.Label(root, text="0.25").place(x=360, y=50)

        # Drowsy Frames Threshold
        tk.Label(root, text="Drowsy Frames Threshold").pack(anchor="w", padx=20, pady=(10, 0))
        self.drowsy_slider = ttk.Scale(root, from_=10, to=60, value=self.settings["DROWSY_FRAMES_THRESHOLD"], orient="horizontal")
        self.drowsy_slider.pack(fill="x", padx=20)
        tk.Label(root, text="10").place(x=20, y=110)
        tk.Label(root, text="60").place(x=360, y=110)

        # Blink Threshold
        tk.Label(root, text="Blink Threshold").pack(anchor="w", padx=20, pady=(10, 0))
        self.blink_slider = ttk.Scale(root, from_=0.1, to=0.3, value=self.settings["BLINK_THRESHOLD"], orient="horizontal")
        self.blink_slider.pack(fill="x", padx=20)
        tk.Label(root, text="0.1").place(x=20, y=170)
        tk.Label(root, text="0.3").place(x=360, y=170)

        # MAR Threshold
        tk.Label(root, text="MAR Threshold").pack(anchor="w", padx=20, pady=(10, 0))
        self.mar_slider = ttk.Scale(root, from_=0.5, to=1.0, value=self.settings.get("MAR_THRESHOLD", 0.6), orient="horizontal")
        self.mar_slider.pack(fill="x", padx=20)
        tk.Label(root, text="0.5").place(x=20, y=230)
        tk.Label(root, text="1.0").place(x=360, y=230)

        # Save Button
        save_button = tk.Button(root, text="Save Settings", command=self.save_settings, bg="blue", fg="white")
        save_button.pack(pady=20)

    def save_settings(self):
        self.settings["EAR_THRESHOLD"] = self.ear_slider.get()
        self.settings["DROWSY_FRAMES_THRESHOLD"] = int(self.drowsy_slider.get())
        self.settings["BLINK_THRESHOLD"] = self.blink_slider.get()
        self.settings["MAR_THRESHOLD"] = self.mar_slider.get()  # Save MAR threshold
        self.on_save(self.settings)

        # Save settings to JSON file
        with open("saved_settings.json", "w") as f:
            json.dump(self.settings, f, indent=4)

        self.root.destroy()
