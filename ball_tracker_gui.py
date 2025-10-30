import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path

class BallTracker:
    def __init__(self, video_path, output_file='ball_positions.json', progress_callback=None):
        self.video_path = video_path
        self.output_file = output_file
        self.positions = []
        self.progress_callback = progress_callback
        self.stop_tracking = False
        
        # HSV range for lime green
        self.lower_green = np.array([35, 100, 100])
        self.upper_green = np.array([85, 255, 255])
    
    def track_ball(self):
        """Process video and track the lime green ball"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            if self.progress_callback:
                self.progress_callback(f"Error: Could not open video {self.video_path}", 0, 0)
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        while True:
            if self.stop_tracking:
                break
                
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps if fps > 0 else frame_count
            
            # Convert to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask for lime green
            mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
            
            # Apply morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Calculate center point
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Store position data
                self.positions.append({
                    'frame': frame_count,
                    'timestamp': timestamp,
                    'center_x': center_x,
                    'center_y': center_y,
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h}
                })
                
                # Draw bounding box and center point
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Add coordinates text
                text = f"({center_x}, {center_y})"
                cv2.putText(frame, text, (center_x + 10, center_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Display frame info
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show the frame
            cv2.imshow('Ball Tracking', frame)
            
            # Update progress
            if self.progress_callback:
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                self.progress_callback(f"Processing frame {frame_count}/{total_frames}", 
                                      progress, len(self.positions))
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_tracking = True
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return True
    
    def save_positions(self):
        """Save tracked positions to JSON file"""
        with open(self.output_file, 'w') as f:
            json.dump(self.positions, f, indent=2)

class VideoProcessingWindow:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Video Processing")
        self.window.geometry("600x300")
        
        self.video_path = tk.StringVar()
        self.json_path = tk.StringVar(value="ball_positions.json")
        self.is_processing = False
        
        self.create_widgets()
    
    def create_widgets(self):
        # Video input section
        frame1 = ttk.LabelFrame(self.window, text="Input Video", padding=10)
        frame1.pack(fill="x", padx=10, pady=5)
        
        ttk.Entry(frame1, textvariable=self.video_path, width=50).pack(side="left", padx=5)
        ttk.Button(frame1, text="Browse...", command=self.browse_video).pack(side="left")
        
        # JSON output section
        frame2 = ttk.LabelFrame(self.window, text="Output JSON", padding=10)
        frame2.pack(fill="x", padx=10, pady=5)
        
        ttk.Entry(frame2, textvariable=self.json_path, width=50).pack(side="left", padx=5)
        ttk.Button(frame2, text="Browse...", command=self.browse_json).pack(side="left")
        
        # Progress section
        frame3 = ttk.LabelFrame(self.window, text="Progress", padding=10)
        frame3.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.status_label = ttk.Label(frame3, text="Ready to process")
        self.status_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(frame3, mode='determinate', length=500)
        self.progress_bar.pack(pady=5)
        
        self.detected_label = ttk.Label(frame3, text="Detected positions: 0")
        self.detected_label.pack(pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(self.window)
        button_frame.pack(pady=10)
        
        self.process_button = ttk.Button(button_frame, text="Process Video", 
                                         command=self.start_processing)
        self.process_button.pack(side="left", padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", 
                                      command=self.stop_processing, state="disabled")
        self.stop_button.pack(side="left", padx=5)
        
        ttk.Button(button_frame, text="Close", command=self.window.destroy).pack(side="left", padx=5)
    
    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.video_path.set(filename)
    
    def browse_json(self):
        filename = filedialog.asksaveasfilename(
            title="Save JSON File",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.json_path.set(filename)
    
    def update_progress(self, message, progress, detected_count):
        self.status_label.config(text=message)
        self.progress_bar['value'] = progress
        self.detected_label.config(text=f"Detected positions: {detected_count}")
        self.window.update()
    
    def start_processing(self):
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file")
            return
        
        if not self.json_path.get():
            messagebox.showerror("Error", "Please specify output JSON file")
            return
        
        self.is_processing = True
        self.process_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self.process_video)
        thread.daemon = True
        thread.start()
    
    def process_video(self):
        try:
            self.tracker = BallTracker(
                self.video_path.get(), 
                self.json_path.get(),
                self.update_progress
            )
            
            success = self.tracker.track_ball()
            
            if success and not self.tracker.stop_tracking:
                self.tracker.save_positions()
                self.update_progress("Processing complete!", 100, len(self.tracker.positions))
                messagebox.showinfo("Success", 
                    f"Video processed successfully!\n"
                    f"Detected {len(self.tracker.positions)} positions\n"
                    f"Saved to {self.json_path.get()}")
            elif self.tracker.stop_tracking:
                self.update_progress("Processing stopped by user", 
                                    self.progress_bar['value'], 
                                    len(self.tracker.positions))
                if self.tracker.positions:
                    self.tracker.save_positions()
                    messagebox.showinfo("Stopped", 
                        f"Processing stopped.\n"
                        f"Detected {len(self.tracker.positions)} positions\n"
                        f"Saved to {self.json_path.get()}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            self.update_progress(f"Error: {str(e)}", 0, 0)
        finally:
            self.is_processing = False
            self.process_button.config(state="normal")
            self.stop_button.config(state="disabled")
    
    def stop_processing(self):
        if hasattr(self, 'tracker'):
            self.tracker.stop_tracking = True

class ViewDataWindow:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("View Data")
        self.window.geometry("800x600")
        
        self.json_path = tk.StringVar()
        self.positions = []
        
        self.create_widgets()
    
    def create_widgets(self):
        # File selection
        frame1 = ttk.LabelFrame(self.window, text="Select JSON File", padding=10)
        frame1.pack(fill="x", padx=10, pady=5)
        
        ttk.Entry(frame1, textvariable=self.json_path, width=50).pack(side="left", padx=5)
        ttk.Button(frame1, text="Browse...", command=self.browse_json).pack(side="left", padx=5)
        ttk.Button(frame1, text="Load & Plot", command=self.load_and_plot).pack(side="left", padx=5)
        
        # Plot area
        self.plot_frame = ttk.Frame(self.window)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=5)
    
    def browse_json(self):
        filename = filedialog.askopenfilename(
            title="Select JSON File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.json_path.set(filename)
    
    def load_and_plot(self):
        if not self.json_path.get():
            messagebox.showerror("Error", "Please select a JSON file")
            return
        
        try:
            with open(self.json_path.get(), 'r') as f:
                self.positions = json.load(f)
            
            if not self.positions:
                messagebox.showwarning("Warning", "No positions found in JSON file")
                return
            
            self.plot_trajectory()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load JSON file:\n{str(e)}")
    
    def plot_trajectory(self):
        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Extract coordinates
        x_coords = [pos['center_x'] for pos in self.positions]
        y_coords = [pos['center_y'] for pos in self.positions]
        frames = [pos['frame'] for pos in self.positions]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. Trajectory plot
        axes[0, 0].plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=1)
        axes[0, 0].scatter(x_coords, y_coords, c=frames, cmap='viridis', s=10)
        axes[0, 0].scatter(x_coords[0], y_coords[0], c='green', s=100, marker='o', label='Start')
        axes[0, 0].scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='X', label='End')
        axes[0, 0].set_xlabel('X Position (pixels)')
        axes[0, 0].set_ylabel('Y Position (pixels)')
        axes[0, 0].set_title('Ball Trajectory')
        axes[0, 0].invert_yaxis()
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. X position over time
        axes[0, 1].plot(frames, x_coords, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('X Position (pixels)')
        axes[0, 1].set_title('Horizontal Position Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Y position over time
        axes[1, 0].plot(frames, y_coords, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Y Position (pixels)')
        axes[1, 0].set_title('Vertical Position Over Time')
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Speed/velocity
        if len(x_coords) > 1:
            speeds = []
            for i in range(1, len(x_coords)):
                dx = x_coords[i] - x_coords[i-1]
                dy = y_coords[i] - y_coords[i-1]
                speed = np.sqrt(dx**2 + dy**2)
                speeds.append(speed)
            
            axes[1, 1].plot(frames[1:], speeds, 'purple', linewidth=2)
            axes[1, 1].set_xlabel('Frame')
            axes[1, 1].set_ylabel('Speed (pixels/frame)')
            axes[1, 1].set_title('Ball Speed Over Time')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Ball Tracker")
        self.root.geometry("400x250")
        
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title = ttk.Label(self.root, text="Ball Tracker", font=("Arial", 20, "bold"))
        title.pack(pady=20)
        
        # Description
        desc = ttk.Label(self.root, text="Track lime green balls in videos", 
                        font=("Arial", 10))
        desc.pack(pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=30)
        
        ttk.Button(button_frame, text="Process Video", 
                  command=self.open_video_processing, 
                  width=20).pack(pady=10)
        
        ttk.Button(button_frame, text="View Data", 
                  command=self.open_view_data, 
                  width=20).pack(pady=10)
        
        ttk.Button(button_frame, text="Exit", 
                  command=self.root.quit, 
                  width=20).pack(pady=10)
    
    def open_video_processing(self):
        VideoProcessingWindow(self.root)
    
    def open_view_data(self):
        ViewDataWindow(self.root)

def main():
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()

if __name__ == '__main__':
    main()
