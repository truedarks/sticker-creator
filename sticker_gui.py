"""
Simple GUI for batch processing images.
Drag and drop images onto the left panel for background removal,
or onto the right panel for sticker creation.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
from pathlib import Path
import threading


def check_and_install_dependencies():
    """Check for required dependencies and offer to install them."""
    missing = []
    
    # Check rembg
    try:
        import rembg
    except ImportError:
        missing.append(("rembg", ["rembg", "onnxruntime"], "Required for background removal"))
    
    # Check tkinterdnd2 (optional but recommended)
    try:
        import tkinterdnd2
    except ImportError:
        missing.append(("tkinterdnd2", ["tkinterdnd2"], "Required for drag-and-drop support"))
    
    if not missing:
        return True
    
    # Show installation dialog
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    msg = "Some required libraries are missing:\n\n"
    for name, pip_name, desc in missing:
        msg += f"• {name} ({desc})\n"
    msg += "\nWould you like to install them now?"
    
    result = messagebox.askyesno(
        "Missing Dependencies",
        msg,
        icon='question'
    )
    
    if not result:
        root.destroy()
        return False
    
    # Install missing packages
    root.destroy()
    install_window = tk.Tk()
    install_window.title("Installing Dependencies")
    install_window.geometry("500x200")
    
    status_label = ttk.Label(
        install_window,
        text="Installing dependencies...\nThis may take a few minutes.",
        font=("Arial", 10),
        justify=tk.CENTER
    )
    status_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
    
    progress = ttk.Progressbar(install_window, mode='indeterminate')
    progress.pack(padx=20, pady=10, fill=tk.X)
    progress.start()
    
    install_window.update()
    
    def install_packages():
        success = True
        for name, pip_packages, desc in missing:
            try:
                status_label.configure(text=f"Installing {name}...")
                install_window.update()
                
                # Install all packages for this dependency
                for package in pip_packages:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", package],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode != 0:
                        success = False
                        print(f"Error installing {package}: {result.stderr}", file=sys.stderr)
            except Exception as e:
                success = False
                print(f"Exception installing {name}: {e}", file=sys.stderr)
        
        progress.stop()
        
        if success:
            status_label.configure(text="[OK] All dependencies installed successfully!\nThe application will restart.")
            install_window.after(2000, install_window.destroy)
        else:
            status_label.configure(
                text="[ERROR] Some installations failed.\nPlease install manually:\npip install rembg onnxruntime tkinterdnd2"
            )
            ttk.Button(
                install_window,
                text="Close",
                command=install_window.destroy
            ).pack(pady=10)
    
    threading.Thread(target=install_packages, daemon=True).start()
    install_window.mainloop()
    
    return True


class StickerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sticker Creator - Batch Processing")
        self.root.geometry("800x400")
        self.root.resizable(False, False)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Drag and drop images to process",
            font=("Arial", 14, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel - Background removal
        left_frame = ttk.LabelFrame(main_frame, text="Background Removal", padding="10")
        left_frame.grid(row=1, column=0, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        left_drop_label = ttk.Label(
            left_frame,
            text="Drop images here\nto remove background",
            font=("Arial", 12),
            foreground="blue",
            cursor="hand2"
        )
        left_drop_label.pack(expand=True, fill=tk.BOTH)
        
        self.left_status = ttk.Label(
            left_frame,
            text="Ready",
            font=("Arial", 9),
            foreground="gray"
        )
        self.left_status.pack(pady=(10, 0))
        
        # Right panel - Sticker creation
        right_frame = ttk.LabelFrame(main_frame, text="Sticker Creation", padding="10")
        right_frame.grid(row=1, column=1, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        right_drop_label = ttk.Label(
            right_frame,
            text="Drop images here\nto create stickers",
            font=("Arial", 12),
            foreground="green",
            cursor="hand2"
        )
        right_drop_label.pack(expand=True, fill=tk.BOTH)
        
        self.right_status = ttk.Label(
            right_frame,
            text="Ready",
            font=("Arial", 9),
            foreground="gray"
        )
        self.right_status.pack(pady=(10, 0))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        
        # Bind drag and drop events
        self.setup_drag_drop(left_frame, left_drop_label, self.left_status, "remove_bg")
        self.setup_drag_drop(right_frame, right_drop_label, self.right_status, "create_sticker")
        
        # Store references
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.left_drop_label = left_drop_label
        self.right_drop_label = right_drop_label
    
    def setup_drag_drop(self, frame, label, status_label, mode):
        """Setup drag and drop functionality for a frame."""
        # Bind click event to browse files
        frame.bind("<Button-1>", lambda e: self.browse_files(mode, status_label))
        label.bind("<Button-1>", lambda e: self.browse_files(mode, status_label))
        
        # Try to use tkinterdnd2 for drag and drop
        try:
            from tkinterdnd2 import DND_FILES
            
            def on_drop(event):
                # Get dropped files
                files = self.root.tk.splitlist(event.data)
                if files:
                    self.process_files(files, mode, status_label)
            
            # Register drop target
            frame.drop_target_register(DND_FILES)
            frame.dnd_bind('<<Drop>>', on_drop)
            label.drop_target_register(DND_FILES)
            label.dnd_bind('<<Drop>>', on_drop)
            
            # Visual feedback on hover
            def on_enter(event):
                frame.configure(relief="sunken")
                label.configure(foreground="darkblue" if mode == "remove_bg" else "darkgreen")
            
            def on_leave(event):
                frame.configure(relief="raised")
                label.configure(foreground="blue" if mode == "remove_bg" else "green")
            
            frame.bind("<Enter>", on_enter)
            frame.bind("<Leave>", on_leave)
            
        except ImportError:
            # Fallback: use file browser button
            browse_btn = ttk.Button(
                frame,
                text="Click to Browse Files",
                command=lambda: self.browse_files(mode, status_label)
            )
            browse_btn.pack(pady=10)
            
            # Show message about drag and drop
            info_label = ttk.Label(
                frame,
                text="(Install tkinterdnd2 for drag & drop)\npip install tkinterdnd2",
                font=("Arial", 8),
                foreground="gray"
            )
            info_label.pack()
    
    def browse_files(self, mode, status_label):
        """Open file browser to select files."""
        from tkinter import filedialog
        
        files = filedialog.askopenfilenames(
            title="Select images to process",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if files:
            self.process_files(files, mode, status_label)
    
    def process_files(self, files, mode, status_label):
        """Process files in background thread."""
        if not files:
            return
        
        # Update status
        file_count = len(files)
        status_label.configure(
            text=f"Processing {file_count} file(s)...",
            foreground="orange"
        )
        status_label.update()
        
        # Process in background thread
        thread = threading.Thread(
            target=self.run_batch_script,
            args=(files, mode, status_label),
            daemon=True
        )
        thread.start()
    
    def run_batch_script(self, files, mode, status_label):
        """Run batch script in background."""
        try:
            # Prepare command
            if mode == "remove_bg":
                script = "batch_remove_bg.py"
                args = ["python", script, "--parallel"] + list(files)
            else:  # create_sticker
                script = "batch_create_sticker.py"
                args = ["python", script, "--parallel"] + list(files)
            
            # Run script
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd()
            )
            
            # Wait for completion
            stdout, stderr = process.communicate()
            
            # Update status
            if process.returncode == 0:
                status_label.after(0, lambda: status_label.configure(
                    text="[OK] Completed successfully!",
                    foreground="green"
                ))
            else:
                status_label.after(0, lambda: status_label.configure(
                    text=f"[ERROR] Error (code: {process.returncode})",
                    foreground="red"
                ))
                if stderr:
                    print(f"Error: {stderr}", file=sys.stderr)
            
            # Reset status after 3 seconds
            self.root.after(3000, lambda: status_label.configure(
                text="Ready",
                foreground="gray"
            ))
            
        except Exception as e:
            status_label.after(0, lambda: status_label.configure(
                text=f"[ERROR] Error: {str(e)[:30]}",
                foreground="red"
            ))
            print(f"Exception: {e}", file=sys.stderr)


def main():
    """Main entry point."""
    # Check and install dependencies on first run
    if not check_and_install_dependencies():
        return
    
    # Check if tkinterdnd2 is available
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except ImportError:
        root = tk.Tk()
    
    app = StickerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

