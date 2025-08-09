import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
from keras.models import load_model
from collections import deque
import threading
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import tempfile
from tkinter import ttk

class ViolenceDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Shop Detection System")
        self.root.geometry("900x700")
        
        # Constants
        self.SEQ_LENGTH = 15
        self.IMG_SIZE = 59
        self.DISPLAY_SIZE = (640, 360)
        self.VIOLENCE_THRESHOLD = 0.5
        self.CONSECUTIVE_VIOLENCE_FRAMES = 10  # Send email after 10 violent frames
        
        # Email configuration (fixed settings)
        self.SMTP_SERVER = "smtp.gmail.com"
        self.SMTP_PORT = 465  # SSL port
        self.SENDER_EMAIL = "teqorbit@gmail.com"
        self.SENDER_PASSWORD = "ctqiajqpduanyweh"  # App-specific password
        self.RECIPIENT_EMAIL = ""
        
        # Processing variables
        self.model = None
        self.cap = None
        self.running = False
        self.processing_complete = False
        self.frame_buffer = deque(maxlen=self.SEQ_LENGTH)
        self.predictions = deque(maxlen=self.SEQ_LENGTH)
        self.current_status = "Ready"
        self.consecutive_violence_count = 0
        self.last_violence_frame = None
        
        # UI Elements
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        self.main_frame = tk.Frame(self.root, padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video display
        self.video_label = tk.Label(self.main_frame, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = tk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Buttons
        self.load_btn = tk.Button(control_frame, text="Load Model", command=self.load_model)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.select_btn = tk.Button(control_frame, text="Select Video", command=self.select_video, state=tk.DISABLED)
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        self.start_btn = tk.Button(control_frame, text="Start", command=self.start_processing, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(control_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Status display
        status_frame = tk.Frame(self.main_frame)
        status_frame.pack(fill=tk.X)
        
        self.status_label = tk.Label(status_frame, text="Status: Ready", font=('Helvetica', 12))
        self.status_label.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.main_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=5)
        
        # Console output
        self.console = tk.Text(self.main_frame, height=8, state=tk.DISABLED)
        self.console.pack(fill=tk.X, pady=10)
        
    def show_toast(self, message):
        toast = tk.Toplevel(self.root)
        toast.geometry("300x50+{}+{}".format(
            self.root.winfo_x() + (self.root.winfo_width() // 2 - 150),
            self.root.winfo_y() + (self.root.winfo_height() // 2 - 25)
        ))
        toast.overrideredirect(1)
        toast.attributes("-alpha", 0.9)
        toast.configure(background='#333333')
        
        label = tk.Label(toast, text=message, fg='white', bg='#333333')
        label.pack(pady=15)
        
        toast.after(3000, toast.destroy)
        
    def log_message(self, message):
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, message + "\n")
        self.console.config(state=tk.DISABLED)
        self.console.see(tk.END)
        
    def send_violence_alert(self, frame):
        try:
            # Save the frame to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.SENDER_EMAIL
            msg['To'] = self.RECIPIENT_EMAIL
            msg['Subject'] = "Violence Detected Alert"
            
            body = "Violence has been detected in 10 consecutive frames of the surveillance footage.\n\n"
            body += "Please review the attached image and take appropriate action."
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach the image
            with open(temp_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', filename="violence_detected.jpg")
                msg.attach(img)
            
            # Send the email in a separate thread
            def send_email_thread():
                try:
                    with smtplib.SMTP_SSL(self.SMTP_SERVER, self.SMTP_PORT) as server:
                        server.login(self.SENDER_EMAIL, self.SENDER_PASSWORD)
                        server.send_message(msg)
                    
                    self.log_message("Violence alert email sent successfully!")
                    self.root.after(0, lambda: self.show_toast("Alert email sent!"))
                except Exception as e:
                    error_msg = f"Failed to send email: {str(e)}"
                    self.log_message(error_msg)
                    self.root.after(0, lambda: self.show_toast(error_msg))
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            threading.Thread(target=send_email_thread, daemon=True).start()
            
        except Exception as e:
            error_msg = f"Error preparing email: {str(e)}"
            self.log_message(error_msg)
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            self.root.after(0, lambda: self.show_toast(error_msg))
        
    def load_model(self):
        self.log_message("Loading violence detection model...")
        self.load_btn.config(state=tk.DISABLED)
        
        def load_model_thread():
            try:
                self.model = load_model('./model.h5')
                self.log_message("Model loaded successfully!")
                self.select_btn.config(state=tk.NORMAL)
            except Exception as e:
                self.log_message(f"Error loading model: {str(e)}")
                self.load_btn.config(state=tk.NORMAL)
                
        threading.Thread(target=load_model_thread, daemon=True).start()
        
    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.video_path = file_path
            self.log_message(f"Selected video: {os.path.basename(file_path)}")
            self.start_btn.config(state=tk.NORMAL)
            
    def start_processing(self):
        if not hasattr(self, 'video_path'):
            return
            
        self.running = True
        self.processing_complete = False
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.select_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)
        
        # Reset counters
        self.frame_buffer = deque(maxlen=self.SEQ_LENGTH)
        self.predictions = deque(maxlen=self.SEQ_LENGTH)
        self.current_status = "Processing..."
        self.consecutive_violence_count = 0
        self.last_violence_frame = None
        
        # Start processing thread
        threading.Thread(target=self.process_video, daemon=True).start()
        
    def stop_processing(self):
        self.running = False
        if not self.processing_complete:
            self.log_message("Processing stopped by user")
        self.enable_controls()
        
    def enable_controls(self):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.select_btn.config(state=tk.NORMAL)
        self.load_btn.config(state=tk.NORMAL)
        
    def process_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.log_message("Error opening video file")
            self.processing_complete = True
            self.root.after(0, self.enable_controls)
            return
            
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        self.progress["maximum"] = total_frames
        self.log_message(f"Starting processing ({total_frames} frames)...")
        
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            try:
                frame_count += 1
                self.progress["value"] = frame_count
                
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display_frame = cv2.resize(frame_rgb, self.DISPLAY_SIZE)
                
                # Prepare frame for model
                model_frame = cv2.resize(frame_rgb, (self.IMG_SIZE, self.IMG_SIZE))
                model_frame = model_frame.astype("float32") / 255.0
                self.frame_buffer.append(model_frame)
                
                # When we have enough frames for a sequence
                if len(self.frame_buffer) == self.SEQ_LENGTH and self.model:
                    # Create sequence tensor
                    sequence = np.expand_dims(np.array(self.frame_buffer), axis=0)
                    
                    # Make prediction
                    pred = self.model.predict(sequence, verbose=0)[0][0]
                    self.predictions.append(pred)
                    
                    print(pred)
                    
                    # Apply temporal smoothing
                    if self.predictions:
                        avg_prediction = np.mean(self.predictions)
                        print(avg_prediction)
                        if avg_prediction > self.VIOLENCE_THRESHOLD:
                            
                            self.current_status = "Shoplifting"
                            self.consecutive_violence_count += 1
                            self.last_violence_frame = frame_rgb
                            
                            # Check if we've reached the threshold for sending an alert
                            if self.consecutive_violence_count >= self.CONSECUTIVE_VIOLENCE_FRAMES:
                                self.send_violence_alert(self.last_violence_frame)
                                self.consecutive_violence_count = 0  # Reset counter
                        else:
                            self.current_status = "No Shoplifting"
                            self.consecutive_violence_count = 0
                        
                # Update UI
                self.update_display(display_frame)
                
                # Control processing speed
                time.sleep(0.03)  # ~30fps
                
            except Exception as e:
                self.log_message(f"Error processing frame {frame_count}: {str(e)}")
                break
                
        # Cleanup
        if self.cap:
            self.cap.release()
        self.running = False
        self.processing_complete = True
        self.log_message(f"Finished processing {frame_count}/{total_frames} frames")
        self.root.after(0, self.enable_controls)
        
    def update_display(self, frame):
        # Add status text to frame
        color = (255, 0, 0) if self.current_status == "Violence" else (0, 255, 0)
        cv2.putText(frame, 
                   f"{self.current_status}", 
                   (35, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, color, 2)
        
        # Convert to PhotoImage
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update status label
        self.status_label.config(text=f"Status: {self.current_status}")
        
        # Update video display
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)
        
    def on_closing(self):
        if self.running:
            if messagebox.askokcancel("Quit", "Processing is still running. Do you want to stop and quit?"):
                self.stop_processing()
                self.root.destroy()
        else:
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ViolenceDetectionUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
