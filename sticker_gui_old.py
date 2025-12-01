"""
Simple GUI for batch processing images.
Drag and drop images onto the left panel for background removal,
or onto the right panel for sticker creation.
"""

# Setup local environment first - MUST be imported before any other imports
from local_env import LOCAL_SITE_PACKAGES
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.absolute()

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import threading
import json


def check_and_install_dependencies():
    """Check for required dependencies and offer to install them."""
    # Setup local environment first
    LOCAL_SITE_PACKAGES.mkdir(parents=True, exist_ok=True)
    if str(LOCAL_SITE_PACKAGES) not in sys.path:
        sys.path.insert(0, str(LOCAL_SITE_PACKAGES))
    
    missing = []
    
    # Check bg_removal module (unified background removal)
    try:
        from bg_removal import get_available_methods
        available_methods = get_available_methods()
        if not available_methods:
            missing.append(("bg_removal", ["rembg", "onnxruntime"], "Required for background removal"))
    except ImportError:
        # Fallback: check rembg directly
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
                
                # Install all packages for this dependency to local directory
                for package in pip_packages:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--target", str(LOCAL_SITE_PACKAGES), package],
                        capture_output=True,
                        text=True,
                        timeout=1800 # 30 minutes
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
        self.root.geometry("800x500")
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
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Method selection frame
        method_frame = ttk.LabelFrame(main_frame, text="Background Removal Method", padding="10")
        method_frame.grid(row=1, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        
        self.method_var = tk.StringVar(value="rembg_gpu")
        
        ttk.Radiobutton(
            method_frame,
            text="rembg GPU ⭐ (Recommended)",
            variable=self.method_var,
            value="rembg_gpu"
        ).grid(row=0, column=0, padx=10, sticky=tk.W)
        
        ttk.Radiobutton(
            method_frame,
            text="rembg CPU",
            variable=self.method_var,
            value="rembg_cpu"
        ).grid(row=0, column=1, padx=10, sticky=tk.W)
        
        ttk.Radiobutton(
            method_frame,
            text="SAM GPU (experimental)",
            variable=self.method_var,
            value="sam_gpu"
        ).grid(row=0, column=2, padx=10, sticky=tk.W)
        
        ttk.Radiobutton(
            method_frame,
            text="SAM CPU (experimental)",
            variable=self.method_var,
            value="sam_cpu"
        ).grid(row=0, column=3, padx=10, sticky=tk.W)
        
        # Method info label
        self.method_info = ttk.Label(
            method_frame,
            text="Select background removal method",
            font=("Arial", 8),
            foreground="gray"
        )
        self.method_info.grid(row=1, column=0, columnspan=4, pady=(5, 0), sticky=tk.W)
        
        # Package size info label
        self.package_size_info = ttk.Label(
            method_frame,
            text="",
            font=("Arial", 8),
            foreground="blue"
        )
        self.package_size_info.grid(row=2, column=0, columnspan=4, pady=(2, 0), sticky=tk.W)
        
        # Buttons frame
        buttons_frame = ttk.Frame(method_frame)
        buttons_frame.grid(row=3, column=0, columnspan=4, pady=(5, 0), sticky=tk.W)
        
        # Install button
        self.install_button = ttk.Button(
            buttons_frame,
            text="Install Required Libraries",
            command=self.install_method_packages,
            state=tk.DISABLED
        )
        self.install_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # LLM Censor settings button (only for SAM methods)
        self.llm_censor_button = ttk.Button(
            buttons_frame,
            text="LLM Censor Settings",
            command=self.open_llm_censor_settings,
            state=tk.DISABLED
        )
        self.llm_censor_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # LLM Censor enabled state
        # Load saved settings or use defaults
        saved_settings = self.load_llm_censor_settings()
        self.llm_censor_enabled = tk.BooleanVar(value=saved_settings.get('enabled', False))
        self.llm_censor_model = tk.StringVar(value=saved_settings.get('model', 'llava:13b'))
        self.llm_censor_url = tk.StringVar(value=saved_settings.get('url', 'http://localhost:11434'))
        self.llm_censor_iterations = tk.IntVar(value=saved_settings.get('iterations', 3))
        self.llm_censor_save_successful = tk.BooleanVar(value=saved_settings.get('save_successful', False))
        
        # Update info when method changes
        self.method_var.trace('w', self.on_method_change)
        self.on_method_change()
        
        # Left panel - Background removal
        left_frame = ttk.LabelFrame(main_frame, text="Background Removal", padding="10")
        left_frame.grid(row=2, column=0, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
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
        right_frame.grid(row=2, column=1, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
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
    
    def get_llm_censor_config_path(self):
        """Get path to LLM censor config file."""
        config_dir = PROJECT_ROOT / ".config"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "llm_censor_config.json"
    
    def load_llm_censor_settings(self):
        """Load LLM censor settings from file."""
        config_path = self.get_llm_censor_config_path()
        if not config_path.exists():
            return {}
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading LLM censor settings: {e}", file=sys.stderr)
            return {}
    
    def save_llm_censor_settings(self):
        """Save LLM censor settings to file."""
        config_path = self.get_llm_censor_config_path()
        try:
            settings = {
                'enabled': self.llm_censor_enabled.get(),
                'model': self.llm_censor_model.get(),
                'url': self.llm_censor_url.get(),
                'iterations': self.llm_censor_iterations.get(),
                'save_successful': self.llm_censor_save_successful.get()
            }
            with open(config_path, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Error saving LLM censor settings: {e}", file=sys.stderr)
    
    def on_method_change(self, *args):
        """Update method info label when method changes."""
        method = self.method_var.get()
        
        # Check method availability
        try:
            from bg_removal import get_method_status, get_method_packages, format_size
            status = get_method_status()
            method_status = status.get(method, {})
            
            is_available = method_status.get('available', False)
            
            if is_available:
                if method == "rembg_cpu":
                    info = "✓ Available - Fast and reliable background removal (CPU)"
                elif method == "rembg_gpu":
                    info = "✓ Available - Best quality with GPU acceleration (Recommended)"
                    # Check if GPU is actually working or falling back to CPU
                    try:
                        import onnxruntime as ort
                        providers = ort.get_available_providers()
                        if 'CUDAExecutionProvider' not in providers:
                            info += " ⚠ (May fall back to CPU - CuDNN missing)"
                        else:
                            # Check if there's a warning in the message
                            msg = method_status.get('message', '')
                            if 'warning' in msg.lower() or 'missing' in msg.lower():
                                info += " ⚠ (Check: May use CPU)"
                    except:
                        pass
                elif method == "sam_cpu":
                    info = "✓ Available - SAM (Experimental, may segment parts separately)"
                    # Show LLM censor status if enabled
                    if self.llm_censor_enabled.get():
                        info += " | LLM Censor: ON"
                elif method == "sam_gpu":
                    info = "✓ Available - SAM with GPU (Experimental, may segment parts separately)"
                    # Check if CUDA is actually available
                    try:
                        import torch
                        if not torch.cuda.is_available():
                            info += " ⚠ (May fall back to CPU)"
                    except:
                        pass
                    # Show LLM censor status if enabled
                    if self.llm_censor_enabled.get():
                        info += " | LLM Censor: ON"
                else:
                    info = method_status.get('message', 'Select background removal method')
            else:
                info = f"⚠ {method_status.get('message', 'Not available')}"
            
            # Get package information - only show if method is not available
            if not is_available:
                packages_info = get_method_packages(method)
                if packages_info['packages']:
                    packages_list = []
                    for pkg_info in packages_info['packages_info']:
                        pkg_name = pkg_info['name']
                        pkg_size = format_size(pkg_info['size_mb'])
                        if pkg_info.get('download', False):
                            packages_list.append(f"  • {pkg_name}: {pkg_size} (download)")
                        else:
                            packages_list.append(f"  • {pkg_name}: {pkg_size}")
                    
                    total_size = format_size(packages_info['total_size_mb'])
                    size_text = f"Required libraries (~{total_size}):\n" + "\n".join(packages_list)
                    if packages_info.get('note'):
                        size_text += f"\n\nNote: {packages_info['note']}"
                    
                    self.package_size_info.configure(text=size_text, foreground="blue")
                    self.install_button.configure(state=tk.NORMAL)
                else:
                    self.package_size_info.configure(text="")
                    self.install_button.configure(state=tk.DISABLED)
            else:
                # Hide requirements when method is available
                self.package_size_info.configure(text="")
                self.install_button.configure(state=tk.DISABLED)
                
        except Exception as e:
            # Fallback to basic info
            if method == "rembg_cpu":
                info = "Uses CPU for background removal (fastest setup)"
            elif method == "rembg_gpu":
                info = "Uses GPU for background removal (requires CUDA and onnxruntime-gpu)"
            elif method == "sam_cpu":
                info = "Uses Segment Anything Model (CPU) (requires torch and segment-anything)"
            elif method == "sam_gpu":
                info = "Uses Segment Anything Model (GPU) (requires CUDA-enabled PyTorch)"
            else:
                info = "Select background removal method"
            
            # Try to get package info even on error
            try:
                from bg_removal import get_method_packages, format_size
                packages_info = get_method_packages(method)
                if packages_info['packages']:
                    packages_list = []
                    for pkg in packages_info['packages']:
                        packages_list.append(f"  • {pkg}")
                    total_size = format_size(packages_info['total_size_mb'])
                    self.package_size_info.configure(
                        text=f"Required libraries (~{total_size}):\n" + "\n".join(packages_list),
                        foreground="blue"
                    )
                    self.install_button.configure(state=tk.NORMAL)
                else:
                    self.package_size_info.configure(text="")
                    self.install_button.configure(state=tk.DISABLED)
            except:
                self.package_size_info.configure(text="")
                self.install_button.configure(state=tk.DISABLED)
        
        self.method_info.configure(text=info)
        
        # Update color based on availability
        if '✓' in info:
            self.method_info.configure(foreground="green")
        elif '⚠' in info:
            self.method_info.configure(foreground="orange")
        else:
            self.method_info.configure(foreground="gray")
        
        # Enable/disable LLM censor button based on method (always check, regardless of availability)
        if method in ('sam_cpu', 'sam_gpu'):
            self.llm_censor_button.configure(state=tk.NORMAL)
        else:
            self.llm_censor_button.configure(state=tk.DISABLED)
    
    def open_llm_censor_settings(self):
        """Open LLM censor settings window."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("LLM Censor Settings")
        settings_window.transient(self.root)
        settings_window.grab_set()
        settings_window.resizable(False, False)
        
        # Main frame with minimal padding
        main_frame = ttk.Frame(settings_window, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="LLM Censor Settings for SAM",
            font=("Arial", 12, "bold")
        )
        title_label.pack(pady=(0, 8))
        
        # Description
        desc_label = ttk.Label(
            main_frame,
            text="LLM censor uses Ollama vision models to improve SAM segmentation quality\nby analyzing images and adjusting parameters iteratively.",
            font=("Arial", 9),
            foreground="gray",
            justify=tk.CENTER
        )
        desc_label.pack(pady=(0, 12))
        
        # Enable checkbox
        enable_check = ttk.Checkbutton(
            main_frame,
            text="Enable LLM Censor",
            variable=self.llm_censor_enabled
        )
        enable_check.pack(anchor=tk.W, pady=(0, 10))
        
        # Model selection
        model_frame = ttk.LabelFrame(main_frame, text="Ollama Model", padding="8")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text="Model Name:").pack(anchor=tk.W)
        model_entry = ttk.Entry(model_frame, textvariable=self.llm_censor_model, width=35)
        model_entry.pack(fill=tk.X, pady=(3, 0))
        
        # Recommended models
        recommended_label = ttk.Label(
            model_frame,
            text="Recommended: llava:13b, llava-next:latest, llava:7b",
            font=("Arial", 8),
            foreground="gray"
        )
        recommended_label.pack(anchor=tk.W, pady=(3, 0))
        
        # Check model button
        def check_model():
            model_name = self.llm_censor_model.get()
            try:
                try:
                    import requests
                except ImportError:
                    messagebox.showerror(
                        "Error",
                        "requests library not installed.\n\nInstall with:\npip install requests"
                    )
                    return
                url = f"{self.llm_censor_url.get()}/api/tags"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    model_names = [m.get('name', '') for m in models]
                    if model_name in model_names:
                        messagebox.showinfo("Model Check", f"✓ Model '{model_name}' is installed and ready!")
                    else:
                        msg = f"Model '{model_name}' not found.\n\nAvailable models:\n" + "\n".join(model_names[:10])
                        if len(model_names) > 10:
                            msg += f"\n... and {len(model_names) - 10} more"
                        msg += f"\n\nWould you like to install '{model_name}'?"
                        if messagebox.askyesno("Model Not Found", msg):
                            install_ollama_model(model_name)
                else:
                    messagebox.showerror("Error", f"Failed to connect to Ollama: {response.status_code}")
            except requests.exceptions.ConnectionError:
                messagebox.showerror("Error", "Cannot connect to Ollama.\n\nMake sure Ollama is running:\nollama serve")
            except Exception as e:
                messagebox.showerror("Error", f"Error checking model: {str(e)}")
        
        check_button = ttk.Button(
            model_frame,
            text="Check / Install Model",
            command=check_model
        )
        check_button.pack(pady=(6, 0))
        
        # Ollama URL
        url_frame = ttk.LabelFrame(main_frame, text="Ollama API URL", padding="8")
        url_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(url_frame, text="URL:").pack(anchor=tk.W)
        url_entry = ttk.Entry(url_frame, textvariable=self.llm_censor_url, width=35)
        url_entry.pack(fill=tk.X, pady=(3, 0))
        
        # Iterations
        iter_frame = ttk.LabelFrame(main_frame, text="Settings", padding="8")
        iter_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(iter_frame, text="Max Iterations:").pack(anchor=tk.W)
        iter_spinbox = ttk.Spinbox(
            iter_frame,
            from_=1,
            to=10,
            textvariable=self.llm_censor_iterations,
            width=10
        )
        iter_spinbox.pack(anchor=tk.W, pady=(3, 0))
        
        ttk.Label(
            iter_frame,
            text="Number of parameter adjustment iterations (1-10)",
            font=("Arial", 8),
            foreground="gray"
        ).pack(anchor=tk.W, pady=(2, 0))
        
        # Save successful parameters checkbox
        save_params_check = ttk.Checkbutton(
            iter_frame,
            text="Save successful parameters for next time",
            variable=self.llm_censor_save_successful
        )
        save_params_check.pack(anchor=tk.W, pady=(8, 0))
        
        ttk.Label(
            iter_frame,
            text="When enabled, parameters that pass AI censor will be saved\nand used as starting point for next image",
            font=("Arial", 8),
            foreground="gray"
        ).pack(anchor=tk.W, pady=(2, 0))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(12, 0))
        
        def save_and_close():
            # Save settings to file
            self.save_llm_censor_settings()
            settings_window.destroy()
            # Update method info to show LLM censor status
            self.on_method_change()
        
        ttk.Button(
            button_frame,
            text="Save",
            command=save_and_close
        ).pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=settings_window.destroy
        ).pack(side=tk.RIGHT)
        
        # Update window size to fit content (no extra space)
        settings_window.update_idletasks()
        width = main_frame.winfo_reqwidth() + 30
        height = main_frame.winfo_reqheight() + 30
        settings_window.geometry(f"{width}x{height}")
        
        def install_ollama_model(model_name):
            """Install Ollama model in a separate window."""
            install_window = tk.Toplevel(settings_window)
            install_window.title("Installing Ollama Model")
            install_window.geometry("500x200")
            install_window.transient(settings_window)
            install_window.grab_set()
            
            status_label = ttk.Label(
                install_window,
                text=f"Installing {model_name}...\nThis may take several minutes.",
                font=("Arial", 10),
                justify=tk.CENTER
            )
            status_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
            
            progress = ttk.Progressbar(install_window, mode='indeterminate')
            progress.pack(padx=20, pady=10, fill=tk.X)
            progress.start()
            
            install_window.update()
            
            def install_model():
                try:
                    # Run ollama pull command
                    result = subprocess.run(
                        ['ollama', 'pull', model_name],
                        capture_output=True,
                        text=True,
                        timeout=1800  # 30 minutes timeout
                    )
                    
                    progress.stop()
                    
                    if result.returncode == 0:
                        status_label.configure(
                            text=f"✓ Successfully installed {model_name}!\n\nYou can now use LLM censor."
                        )
                        ttk.Button(
                            install_window,
                            text="Close",
                            command=install_window.destroy
                        ).pack(pady=10)
                    else:
                        error_msg = result.stderr or result.stdout or "Unknown error"
                        status_label.configure(
                            text=f"✗ Installation failed:\n\n{error_msg[:200]}"
                        )
                        ttk.Button(
                            install_window,
                            text="Close",
                            command=install_window.destroy
                        ).pack(pady=10)
                except subprocess.TimeoutExpired:
                    progress.stop()
                    status_label.configure(
                        text="✗ Installation timeout.\n\nModel may be very large.\nTry installing manually:\nollama pull " + model_name
                    )
                    ttk.Button(
                        install_window,
                        text="Close",
                        command=install_window.destroy
                    ).pack(pady=10)
                except FileNotFoundError:
                    progress.stop()
                    status_label.configure(
                        text="✗ Ollama not found in PATH.\n\nMake sure Ollama is installed and\navailable in system PATH."
                    )
                    ttk.Button(
                        install_window,
                        text="Close",
                        command=install_window.destroy
                    ).pack(pady=10)
                except Exception as e:
                    progress.stop()
                    status_label.configure(
                        text=f"✗ Error: {str(e)}"
                    )
                    ttk.Button(
                        install_window,
                        text="Close",
                        command=install_window.destroy
                    ).pack(pady=10)
            
            threading.Thread(target=install_model, daemon=True).start()
            install_window.mainloop()
    
    def restart_app(self, window=None):
        """Restart the application."""
        if window:
            window.destroy()
        self.root.destroy()
        script_path = os.path.abspath(__file__)
        subprocess.Popen([sys.executable, script_path], cwd=os.getcwd())
    
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
    
    def install_method_packages(self):
        """Install required packages for selected method."""
        method = self.method_var.get()
        
        try:
            from bg_removal import get_method_packages, format_size, check_cuda_installed, detect_nvidia_gpu
            
            # Check CUDA requirements for GPU methods
            if method in ('rembg_gpu', 'sam_gpu'):
                cuda_installed, cuda_info = check_cuda_installed()
                gpu_info = detect_nvidia_gpu()
                
                # Build warning message
                warnings = []
                
                # Check CUDA Toolkit
                if not cuda_installed:
                    warnings.append("⚠ CUDA Toolkit not found")
                else:
                    # CUDA is installed, but check for additional requirements
                    if method == 'rembg_gpu':
                        # Check for CuDNN (needed for onnxruntime-gpu)
                        if isinstance(cuda_info, dict):
                            if not cuda_info.get('cudnn_ok', False):
                                cuda_version = cuda_info.get('version', 'unknown')
                                warnings.append(
                                    f"⚠ CuDNN not found for CUDA {cuda_version}\n"
                                    "   onnxruntime-gpu may fall back to CPU without CuDNN"
                                )
                
                # Check GPU
                if not gpu_info.get('detected'):
                    warnings.append("⚠ NVIDIA GPU not detected")
                
                # Show warnings if any
                if warnings:
                    cuda_msg = "GPU Setup Warning:\n\n" + "\n\n".join(warnings) + "\n\n"
                    
                    # Add recommendations
                    if not cuda_installed:
                        if gpu_info.get('detected'):
                            cuda_msg += f"✓ GPU Detected: {gpu_info.get('name', 'Unknown')}\n\n"
                            cuda_msg += (
                                "To enable GPU acceleration:\n"
                                "1. Download CUDA Toolkit:\n"
                                "   https://developer.nvidia.com/cuda-downloads\n"
                                "2. Recommended versions:\n"
                                "   • CUDA 11.8 (best compatibility)\n"
                                "   • CUDA 12.1 (newer, may need CuDNN)\n"
                                "3. Install CUDA Toolkit\n"
                                "4. Restart this application\n\n"
                            )
                        else:
                            cuda_msg += (
                                "GPU methods require:\n"
                                "• NVIDIA GPU with CUDA support\n"
                                "• Latest NVIDIA drivers\n"
                                "• CUDA Toolkit (11.8 or 12.x)\n\n"
                            )
                    else:
                        # CUDA installed but missing CuDNN
                        if method == 'rembg_gpu' and isinstance(cuda_info, dict):
                            if not cuda_info.get('cudnn_ok', False):
                                cuda_msg += (
                                    "CuDNN is required for onnxruntime-gpu:\n"
                                    "1. Download CuDNN from:\n"
                                    "   https://developer.nvidia.com/cudnn\n"
                                    "2. Extract and copy DLLs to CUDA bin folder\n\n"
                                    "OR:\n"
                                    "• Use rembg CPU method (works fine)\n"
                                    "• Install compatible CUDA version\n\n"
                                )
                    
                    if isinstance(cuda_info, dict):
                        cuda_msg += f"System: {cuda_info.get('message', 'CUDA not installed')}\n\n"
                    
                    cuda_msg += (
                        "Continue with library installation?\n"
                        "(GPU may not work without proper CUDA setup)"
                    )
                    
                    result_cuda = messagebox.askyesno(
                        "GPU Setup Warning",
                        cuda_msg,
                        icon='warning'
                    )
                    if not result_cuda:
                        return
            
            packages_info = get_method_packages(method)
            
            if not packages_info['packages']:
                messagebox.showinfo("Info", "No additional libraries required for this method.")
                return
            
            packages = packages_info['packages']
            total_size = format_size(packages_info['total_size_mb'])
            
            # Show confirmation dialog
            packages_list = "\n".join([f"  • {pkg}" for pkg in packages])
            note = packages_info.get('note', '')
            note_text = f"\n\nNote: {note}" if note else ""
            
            msg = (
                f"Will install {len(packages)} library(ies) (~{total_size}):\n\n"
                f"{packages_list}"
                f"{note_text}\n\n"
                f"All libraries will be installed locally in the project folder.\n"
                f"Continue installation?"
            )
            
            result = messagebox.askyesno(
                "Install Libraries",
                msg,
                icon='question'
            )
            
            if not result:
                return
            
            # Show installation window
            install_window = tk.Toplevel(self.root)
            install_window.title("Installing Libraries")
            install_window.geometry("500x300")
            install_window.transient(self.root)
            install_window.grab_set()
            
            status_label = ttk.Label(
                install_window,
                text=f"Installing libraries for {method}...\nThis may take several minutes.",
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
                errors = []
                installed_torchvision = False
                installed_packages = []
                
                # Ensure local site-packages directory exists
                LOCAL_SITE_PACKAGES.mkdir(parents=True, exist_ok=True)
                
                # PRE-INSTALLATION: Handle package conflicts
                if method == 'rembg_gpu':
                    status_label.configure(text="Removing conflicting packages...")
                    install_window.update()
                    # Remove BOTH onnxruntime and onnxruntime-gpu to avoid conflicts
                    subprocess.run(
                        [sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"],
                        capture_output=True,
                        timeout=120
                    )
                    print("Removed conflicting onnxruntime packages", file=sys.stderr)
                
                if method == 'sam_gpu':
                    status_label.configure(text="Removing CPU-only PyTorch...")
                    install_window.update()
                    # Remove CPU version of torch/torchvision
                    subprocess.run(
                        [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision"],
                        capture_output=True,
                        timeout=120
                    )
                    print("Removed CPU-only PyTorch", file=sys.stderr)
                
                for i, package in enumerate(packages):
                    # Skip torchvision if already installed with torch (SAM GPU)
                    if method == 'sam_gpu' and package == 'torchvision' and installed_torchvision:
                        continue
                    
                    try:
                        
                        status_label.configure(text=f"Installing {package} ({i+1}/{len(packages)})...")
                        install_window.update()
                        
                        # Special handling for torch with CUDA (SAM GPU)
                        if method == 'sam_gpu' and package == 'torch':
                            # Detect CUDA version and install matching PyTorch
                            cuda_installed, cuda_info = check_cuda_installed()
                            if cuda_installed and isinstance(cuda_info, dict):
                                from bg_removal import get_cuda_version_for_pytorch
                                cuda_version = cuda_info.get('version', '11.8')
                                pytorch_cuda = get_cuda_version_for_pytorch(cuda_version)
                                status_label.configure(
                                    text=f"Installing torch with CUDA {cuda_version} support...\n(This may take several minutes)"
                                )
                                install_window.update()
                                cmd = [
                                    sys.executable, "-m", "pip", "install", 
                                    "--target", str(LOCAL_SITE_PACKAGES),
                                    "torch", "torchvision", "--index-url", 
                                    f"https://download.pytorch.org/whl/{pytorch_cuda}"
                                ]
                            else:
                                # Try CUDA 11.8 as default
                                status_label.configure(
                                    text=f"Installing torch with CUDA support...\n(This may take several minutes)"
                                )
                                install_window.update()
                                cmd = [
                                    sys.executable, "-m", "pip", "install", 
                                    "--target", str(LOCAL_SITE_PACKAGES),
                                    "torch", "torchvision", "--index-url", 
                                    "https://download.pytorch.org/whl/cu118"
                                ]
                            installed_torchvision = True
                        else:
                            cmd = [
                                sys.executable, "-m", "pip", "install", 
                                "--target", str(LOCAL_SITE_PACKAGES), package
                            ]
                        
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=3600  # 60 minutes timeout for large downloads (PyTorch)
                        )
                        
                        if result.returncode != 0:
                            # For SAM GPU torch, try CPU version as fallback
                            if method == 'sam_gpu' and package == 'torch':
                                status_label.configure(
                                    text=f"CUDA version of torch unavailable.\nInstalling CPU version..."
                                )
                                install_window.update()
                                cmd_cpu = [
                                    sys.executable, "-m", "pip", "install", 
                                    "--target", str(LOCAL_SITE_PACKAGES), "torch", "torchvision"
                                ]
                                result = subprocess.run(
                                    cmd_cpu,
                                    capture_output=True,
                                    text=True,
                                    timeout=3600
                                )
                                if result.returncode != 0:
                                    success = False
                                    errors.append(f"{package}: {result.stderr[:200]}")
                                    print(f"Error installing {package}: {result.stderr}", file=sys.stderr)
                                else:
                                    installed_torchvision = True
                                    installed_packages.extend(['torch', 'torchvision'])
                            else:
                                success = False
                                errors.append(f"{package}: {result.stderr[:200]}")
                                print(f"Error installing {package}: {result.stderr}", file=sys.stderr)
                        else:
                            installed_packages.append(package)
                            if method == 'sam_gpu' and package == 'torch':
                                installed_packages.append('torchvision')
                            
                    except subprocess.TimeoutExpired:
                        success = False
                        errors.append(f"{package}: Timeout (exceeded time limit)")
                    except Exception as e:
                        success = False
                        errors.append(f"{package}: {str(e)[:200]}")
                
                # Setup environment after installation
                method_now_available = False
                if success and installed_packages:
                    try:
                        status_label.configure(text="Configuring environment...")
                        install_window.update()
                        
                        # Reload local environment
                        from local_env import setup_local_environment
                        setup_local_environment()
                        
                        # Setup CUDA environment if GPU packages installed
                        if method in ('rembg_gpu', 'sam_gpu'):
                            from bg_removal import setup_cuda_environment
                            cuda_setup = setup_cuda_environment()
                            if cuda_setup:
                                print("CUDA environment configured successfully", file=sys.stderr)
                        
                        # Reload libraries and update availability flags
                        status_label.configure(text="Verifying installation...")
                        install_window.update()
                        
                        # Wait a moment for file system to sync
                        import time
                        time.sleep(0.5)
                        
                        from bg_removal import reload_libraries, get_method_status
                        reload_libraries()
                        
                        # Wait a bit more and check again
                        time.sleep(0.5)
                        
                        # Check if method is now available (force reload to get latest status)
                        status = get_method_status(force_reload=True)
                        method_status = status.get(method, {})
                        method_now_available = method_status.get('available', False)
                        
                        if method_now_available:
                            print(f"Method {method} is now available!", file=sys.stderr)
                            
                            # ADDITIONAL CHECK: For GPU methods, verify it's not falling back to CPU
                            if method == 'rembg_gpu':
                                try:
                                    import onnxruntime as ort
                                    providers = ort.get_available_providers()
                                    if 'CUDAExecutionProvider' not in providers:
                                        print("WARNING: CUDAExecutionProvider not found!", file=sys.stderr)
                                        method_now_available = False
                                        method_status['message'] = "onnxruntime-gpu installed but CUDA provider not available. May need CuDNN."
                                    else:
                                        print(f"✓ CUDA provider available: {providers}", file=sys.stderr)
                                except Exception as e:
                                    print(f"Warning checking CUDA provider: {e}", file=sys.stderr)
                            
                            elif method == 'sam_gpu':
                                try:
                                    import torch
                                    if not torch.cuda.is_available():
                                        print("WARNING: torch.cuda.is_available() = False!", file=sys.stderr)
                                        method_now_available = False
                                        method_status['message'] = "PyTorch installed but CUDA not available. Check CUDA Toolkit installation."
                                    else:
                                        print(f"✓ PyTorch CUDA available: {torch.version.cuda}", file=sys.stderr)
                                        print(f"✓ GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)
                                except Exception as e:
                                    print(f"Warning checking PyTorch CUDA: {e}", file=sys.stderr)
                        else:
                            print(f"Method {method} status: {method_status.get('message', 'Unknown')}", file=sys.stderr)
                            # Try one more time after a delay
                            time.sleep(1)
                            reload_libraries()
                            status = get_method_status(force_reload=True)
                            method_status = status.get(method, {})
                            method_now_available = method_status.get('available', False)
                        
                    except Exception as e:
                        # Not critical, continue
                        print(f"Environment setup warning: {e}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()
                
                progress.stop()
                
                if success:
                    # Prepare status message based on verification results
                    if method_now_available:
                        final_message = "[OK] All libraries installed successfully!\n\n✓ Method is now available!"
                        
                        # Add GPU-specific warnings if needed
                        if method == 'rembg_gpu':
                            try:
                                import onnxruntime as ort
                                providers = ort.get_available_providers()
                                if 'CUDAExecutionProvider' not in providers:
                                    final_message += "\n\n⚠ WARNING: CUDA provider not detected."
                                    final_message += "\nMethod may fall back to CPU."
                                    final_message += "\nYou may need to install CuDNN."
                            except:
                                pass
                        
                        elif method == 'sam_gpu':
                            try:
                                import torch
                                if not torch.cuda.is_available():
                                    final_message += "\n\n⚠ WARNING: CUDA not available in PyTorch."
                                    final_message += "\nMethod may not use GPU."
                                    final_message += "\nCheck CUDA Toolkit installation."
                            except:
                                pass
                        
                        final_message += "\n\nApplication will restart..."
                    else:
                        final_message = "[OK] Libraries installed.\n\n"
                        
                        # Show specific issues
                        if method in ('rembg_gpu', 'sam_gpu'):
                            final_message += "⚠ GPU support verification failed.\n"
                            if method_status:
                                issue_msg = method_status.get('message', '')
                                if issue_msg:
                                    # Truncate long messages
                                    if len(issue_msg) > 100:
                                        issue_msg = issue_msg[:100] + "..."
                                    final_message += f"\nIssue: {issue_msg}\n"
                            
                            final_message += "\nLibraries are installed but GPU may not work."
                            final_message += "\nCheck documentation for GPU setup."
                        
                        final_message += "\n\nApplication will restart..."
                    
                    status_label.configure(text=final_message)
                    install_window.update()
                    
                    # Add test button for GPU methods
                    if method in ('rembg_gpu', 'sam_gpu') and success:
                        button_frame = ttk.Frame(install_window)
                        button_frame.pack(pady=10)
                        
                        def run_performance_test():
                            install_window.destroy()
                            self.root.destroy()
                            # Run performance test in new terminal
                            test_script = os.path.join(os.getcwd(), "test_performance.py")
                            if os.path.exists(test_script):
                                subprocess.Popen([sys.executable, test_script])
                            # Then restart app
                            script_path = os.path.abspath(__file__)
                            subprocess.Popen([sys.executable, script_path], cwd=os.getcwd())
                        
                        ttk.Button(
                            button_frame,
                            text="Run Performance Test",
                            command=run_performance_test
                        ).pack(side=tk.LEFT, padx=5)
                        
                        ttk.Button(
                            button_frame,
                            text="Restart App",
                            command=lambda: restart_app()
                        ).pack(side=tk.LEFT, padx=5)
                        
                        # Don't auto-restart if showing test button
                        def restart_app():
                            install_window.destroy()
                            self.root.destroy()
                            script_path = os.path.abspath(__file__)
                            subprocess.Popen([sys.executable, script_path], cwd=os.getcwd())
                    else:
                        # Wait a bit, then restart
                        def restart_app():
                            install_window.destroy()
                            # Close current window first
                            self.root.destroy()
                            # Restart by launching new process
                            script_path = os.path.abspath(__file__)
                            subprocess.Popen([sys.executable, script_path], cwd=os.getcwd())
                        
                        install_window.after(3000, restart_app)  # Увеличено до 3 секунд чтобы прочитать
                else:
                    error_msg = "\n".join(errors[:5])  # Show first 5 errors
                    if len(errors) > 5:
                        error_msg += f"\n... and {len(errors) - 5} more errors"
                    
                    status_label.configure(
                        text=f"[ERROR] Some installations failed.\n\n{error_msg}\n\nTry installing manually via command line."
                    )
                    ttk.Button(
                        install_window,
                        text="Close",
                        command=install_window.destroy
                    ).pack(pady=10)
            
            threading.Thread(target=install_packages, daemon=True).start()
            install_window.mainloop()
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to get package information:\n{str(e)}"
            )
    
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
            # Get selected method
            method = self.method_var.get()
            
            # Prepare command
            if mode == "remove_bg":
                script = "batch_remove_bg.py"
                args = ["python", script, "--method", method] + list(files)
                
                # Add LLM censor parameters if enabled and method is SAM
                if self.llm_censor_enabled.get() and method in ('sam_cpu', 'sam_gpu'):
                    args.extend([
                        "--llm-censor",
                        "--llm-model", self.llm_censor_model.get(),
                        "--llm-url", self.llm_censor_url.get(),
                        "--llm-iterations", str(self.llm_censor_iterations.get())
                    ])
                    # Add save successful params if enabled
                    if self.llm_censor_save_successful.get():
                        args.append("--save-successful-params")
                else:
                    # Only use parallel if LLM censor is not enabled
                    args.insert(2, "--parallel")
            else:  # create_sticker
                script = "batch_create_sticker.py"
                args = ["python", script, "--parallel", "--method", method] + list(files)
            
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

