"""
Federated Learning Client - Dataset Selection GUI
==================================================
Modern dark-themed tkinter GUI for FL client training.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import subprocess
import threading
import os
from pathlib import Path


class DarkTheme:
    """Dark theme color palette"""
    BG_DARK = "#1a1a2e"
    BG_MEDIUM = "#16213e"
    BG_LIGHT = "#0f3460"
    BG_CARD = "#1f2937"

    FG_PRIMARY = "#e4e4e7"
    FG_SECONDARY = "#a1a1aa"
    FG_MUTED = "#71717a"

    ACCENT_PRIMARY = "#6366f1"  # Indigo
    ACCENT_SUCCESS = "#22c55e"  # Green
    ACCENT_WARNING = "#f59e0b"  # Amber
    ACCENT_DANGER = "#ef4444"   # Red
    ACCENT_INFO = "#3b82f6"     # Blue

    ENTRY_BG = "#27272a"
    ENTRY_FG = "#e4e4e7"

    BUTTON_BG = "#6366f1"
    BUTTON_FG = "#ffffff"
    BUTTON_HOVER = "#4f46e5"


class FederatedLearningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FL Client - Federated Learning Training")
        self.root.geometry("1000x750")
        self.root.resizable(True, True)
        self.root.configure(bg=DarkTheme.BG_DARK)

        # Variables
        self.dataset_path = tk.StringVar()
        self.epochs = tk.IntVar(value=10)
        self.batch_size = tk.IntVar(value=8)
        self.learning_rate = tk.DoubleVar(value=0.00001)
        self.client_id = tk.StringVar(value="client1")
        self.client_name = tk.StringVar(value="Client 1")
        self.training_running = False
        self.process = None

        # Apply dark theme
        self.setup_styles()

        # Create GUI elements
        self.create_widgets()

        # Set default path if available from .env
        self.load_default_path()

    def setup_styles(self):
        """Configure ttk styles for dark theme"""
        style = ttk.Style()
        style.theme_use('clam')

        # Frame styles
        style.configure("Dark.TFrame", background=DarkTheme.BG_DARK)
        style.configure("Card.TFrame", background=DarkTheme.BG_CARD)

        # Label styles
        style.configure("Dark.TLabel",
                       background=DarkTheme.BG_DARK,
                       foreground=DarkTheme.FG_PRIMARY,
                       font=("Segoe UI", 10))

        style.configure("Title.TLabel",
                       background=DarkTheme.BG_DARK,
                       foreground=DarkTheme.FG_PRIMARY,
                       font=("Segoe UI", 20, "bold"))

        style.configure("Subtitle.TLabel",
                       background=DarkTheme.BG_DARK,
                       foreground=DarkTheme.FG_SECONDARY,
                       font=("Segoe UI", 10))

        style.configure("Card.TLabel",
                       background=DarkTheme.BG_CARD,
                       foreground=DarkTheme.FG_PRIMARY,
                       font=("Segoe UI", 10))

        style.configure("CardTitle.TLabel",
                       background=DarkTheme.BG_CARD,
                       foreground=DarkTheme.FG_PRIMARY,
                       font=("Segoe UI", 11, "bold"))

        style.configure("Muted.TLabel",
                       background=DarkTheme.BG_CARD,
                       foreground=DarkTheme.FG_MUTED,
                       font=("Segoe UI", 9))

        # Entry styles
        style.configure("Dark.TEntry",
                       fieldbackground=DarkTheme.ENTRY_BG,
                       foreground=DarkTheme.ENTRY_FG,
                       insertcolor=DarkTheme.FG_PRIMARY)

        # Button styles
        style.configure("Accent.TButton",
                       background=DarkTheme.ACCENT_PRIMARY,
                       foreground=DarkTheme.BUTTON_FG,
                       font=("Segoe UI", 10, "bold"),
                       padding=(20, 10))
        style.map("Accent.TButton",
                 background=[("active", DarkTheme.BUTTON_HOVER)])

        style.configure("Success.TButton",
                       background=DarkTheme.ACCENT_SUCCESS,
                       foreground=DarkTheme.BUTTON_FG,
                       font=("Segoe UI", 10, "bold"),
                       padding=(20, 10))

        style.configure("Danger.TButton",
                       background=DarkTheme.ACCENT_DANGER,
                       foreground=DarkTheme.BUTTON_FG,
                       font=("Segoe UI", 10, "bold"),
                       padding=(20, 10))

        style.configure("Secondary.TButton",
                       background=DarkTheme.BG_LIGHT,
                       foreground=DarkTheme.FG_PRIMARY,
                       font=("Segoe UI", 10),
                       padding=(15, 8))

        # Spinbox style
        style.configure("Dark.TSpinbox",
                       fieldbackground=DarkTheme.ENTRY_BG,
                       foreground=DarkTheme.ENTRY_FG,
                       background=DarkTheme.BG_LIGHT)

        # LabelFrame style
        style.configure("Card.TLabelframe",
                       background=DarkTheme.BG_CARD,
                       foreground=DarkTheme.FG_PRIMARY)
        style.configure("Card.TLabelframe.Label",
                       background=DarkTheme.BG_CARD,
                       foreground=DarkTheme.ACCENT_PRIMARY,
                       font=("Segoe UI", 11, "bold"))

    def create_widgets(self):
        """Create all GUI widgets"""

        # Main container
        main_frame = ttk.Frame(self.root, style="Dark.TFrame", padding="20")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

        # Header Section
        header_frame = ttk.Frame(main_frame, style="Dark.TFrame")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))

        title_label = ttk.Label(
            header_frame,
            text="Federated Learning Client",
            style="Title.TLabel"
        )
        title_label.pack(anchor="w")

        subtitle_label = ttk.Label(
            header_frame,
            text="Train and upload model weights to the FL server",
            style="Subtitle.TLabel"
        )
        subtitle_label.pack(anchor="w", pady=(5, 0))

        # Dataset Selection Card
        dataset_card = ttk.LabelFrame(
            main_frame,
            text="  Dataset Selection  ",
            style="Card.TLabelframe",
            padding="15"
        )
        dataset_card.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        dataset_card.columnconfigure(1, weight=1)

        ttk.Label(dataset_card, text="Dataset Path:", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=5
        )

        # Path entry with dark styling
        self.path_entry = tk.Entry(
            dataset_card,
            textvariable=self.dataset_path,
            font=("Segoe UI", 10),
            bg=DarkTheme.ENTRY_BG,
            fg=DarkTheme.ENTRY_FG,
            insertbackground=DarkTheme.FG_PRIMARY,
            relief="flat",
            highlightthickness=1,
            highlightbackground=DarkTheme.BG_LIGHT,
            highlightcolor=DarkTheme.ACCENT_PRIMARY
        )
        self.path_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5, ipady=8)

        browse_btn = ttk.Button(
            dataset_card,
            text="Browse",
            command=self.browse_dataset,
            style="Secondary.TButton"
        )
        browse_btn.grid(row=0, column=2, padx=(10, 0), pady=5)

        ttk.Label(
            dataset_card,
            text="Select folder with train/, valid/, test/ subfolders containing Benign/ and Malignant/ classes",
            style="Muted.TLabel"
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))

        # Training Parameters Card
        params_card = ttk.LabelFrame(
            main_frame,
            text="  Training Parameters  ",
            style="Card.TLabelframe",
            padding="15"
        )
        params_card.grid(row=2, column=0, sticky="ew", pady=(0, 15))

        # Create a grid for parameters
        params_inner = ttk.Frame(params_card, style="Card.TFrame")
        params_inner.pack(fill="x")

        # Row 1: Epochs and Batch Size
        self.create_param_field(params_inner, "Epochs:", self.epochs, 0, 0, spinbox=True, from_=1, to=100)
        self.create_param_field(params_inner, "Batch Size:", self.batch_size, 0, 2, spinbox=True, from_=1, to=64)

        # Row 2: Learning Rate
        self.create_param_field(params_inner, "Learning Rate:", self.learning_rate, 1, 0, width=15)

        hint_label = ttk.Label(
            params_inner,
            text="Recommended: 1e-5 for fine-tuning",
            style="Muted.TLabel"
        )
        hint_label.grid(row=1, column=2, columnspan=2, sticky="w", padx=20)

        # Row 3: Client Info
        self.create_param_field(params_inner, "Client ID:", self.client_id, 2, 0, width=15)
        self.create_param_field(params_inner, "Client Name:", self.client_name, 2, 2, width=20)

        # Control Buttons
        button_frame = ttk.Frame(main_frame, style="Dark.TFrame")
        button_frame.grid(row=3, column=0, sticky="ew", pady=(0, 15))

        self.start_btn = ttk.Button(
            button_frame,
            text="▶  Start Training",
            command=self.start_training,
            style="Success.TButton"
        )
        self.start_btn.pack(side="left", padx=(0, 10))

        self.stop_btn = ttk.Button(
            button_frame,
            text="■  Stop",
            command=self.stop_training,
            style="Danger.TButton",
            state=tk.DISABLED
        )
        self.stop_btn.pack(side="left", padx=(0, 10))

        clear_btn = ttk.Button(
            button_frame,
            text="Clear Log",
            command=self.clear_log,
            style="Secondary.TButton"
        )
        clear_btn.pack(side="left")

        # Server status indicator
        self.server_status = ttk.Label(
            button_frame,
            text="",
            style="Dark.TLabel"
        )
        self.server_status.pack(side="right")

        # Log Output Card
        log_card = ttk.LabelFrame(
            main_frame,
            text="  Training Log  ",
            style="Card.TLabelframe",
            padding="10"
        )
        log_card.grid(row=4, column=0, sticky="nsew", pady=(0, 10))
        log_card.columnconfigure(0, weight=1)
        log_card.rowconfigure(0, weight=1)

        # Log text with dark theme
        self.log_text = tk.Text(
            log_card,
            height=15,
            font=("Consolas", 10),
            bg="#0d1117",
            fg="#c9d1d9",
            insertbackground="#c9d1d9",
            relief="flat",
            highlightthickness=0,
            wrap=tk.WORD,
            padx=10,
            pady=10
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        # Scrollbar for log
        scrollbar = ttk.Scrollbar(log_card, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # Status Bar
        status_frame = ttk.Frame(main_frame, style="Dark.TFrame")
        status_frame.grid(row=5, column=0, sticky="ew")

        self.status_var = tk.StringVar(value="Ready")
        self.status_indicator = tk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 10),
            bg=DarkTheme.BG_DARK,
            fg=DarkTheme.FG_SECONDARY,
            anchor="w"
        )
        self.status_indicator.pack(side="left", fill="x", expand=True)

        # GPU indicator
        self.gpu_label = tk.Label(
            status_frame,
            text="GPU: Checking...",
            font=("Segoe UI", 9),
            bg=DarkTheme.BG_DARK,
            fg=DarkTheme.FG_MUTED
        )
        self.gpu_label.pack(side="right")

        # Check GPU status
        self.check_gpu_status()

    def create_param_field(self, parent, label, variable, row, col, spinbox=False, width=10, **kwargs):
        """Create a parameter input field"""
        ttk.Label(parent, text=label, style="Card.TLabel").grid(
            row=row, column=col, sticky="w", padx=(0, 5), pady=8
        )

        if spinbox:
            entry = tk.Spinbox(
                parent,
                textvariable=variable,
                width=width,
                font=("Segoe UI", 10),
                bg=DarkTheme.ENTRY_BG,
                fg=DarkTheme.ENTRY_FG,
                buttonbackground=DarkTheme.BG_LIGHT,
                relief="flat",
                highlightthickness=1,
                highlightbackground=DarkTheme.BG_LIGHT,
                highlightcolor=DarkTheme.ACCENT_PRIMARY,
                from_=kwargs.get('from_', 1),
                to=kwargs.get('to', 100)
            )
        else:
            entry = tk.Entry(
                parent,
                textvariable=variable,
                width=width,
                font=("Segoe UI", 10),
                bg=DarkTheme.ENTRY_BG,
                fg=DarkTheme.ENTRY_FG,
                insertbackground=DarkTheme.FG_PRIMARY,
                relief="flat",
                highlightthickness=1,
                highlightbackground=DarkTheme.BG_LIGHT,
                highlightcolor=DarkTheme.ACCENT_PRIMARY
            )

        entry.grid(row=row, column=col+1, sticky="w", padx=(0, 20), pady=8, ipady=5)
        return entry

    def check_gpu_status(self):
        """Check if GPU is available"""
        try:
            import subprocess
            result = subprocess.run(
                ['python', '-c', 'import tensorflow as tf; gpus = tf.config.list_physical_devices("GPU"); print(len(gpus))'],
                capture_output=True, text=True, timeout=10
            )
            gpu_count = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
            if gpu_count > 0:
                self.gpu_label.config(text=f"GPU: Available ({gpu_count})", fg=DarkTheme.ACCENT_SUCCESS)
            else:
                self.gpu_label.config(text="GPU: Not detected (CPU mode)", fg=DarkTheme.ACCENT_WARNING)
        except:
            self.gpu_label.config(text="GPU: Unknown", fg=DarkTheme.FG_MUTED)

    def load_default_path(self):
        """Load default dataset path from .env if available"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            default_path = os.getenv('LOCAL_DATASET_PATH', '')
            if default_path and os.path.isdir(default_path):
                self.dataset_path.set(default_path)
                self.log("✅ Loaded default dataset path from .env")
        except Exception:
            pass

    def browse_dataset(self):
        """Open folder browser to select dataset directory"""
        folder_path = filedialog.askdirectory(
            title="Select Dataset Folder",
            initialdir=self.dataset_path.get() or os.path.expanduser("~")
        )

        if folder_path:
            self.dataset_path.set(folder_path)
            self.validate_dataset_structure(folder_path)

    def validate_dataset_structure(self, path):
        """Validate that the selected folder has the correct structure"""
        required_folders = ['train', 'valid', 'test']
        path_obj = Path(path)

        missing_folders = []
        for folder in required_folders:
            if not (path_obj / folder).is_dir():
                missing_folders.append(folder)

        if missing_folders:
            self.log(f"⚠️ Warning: Missing folders: {', '.join(missing_folders)}")
            messagebox.showwarning(
                "Incomplete Dataset Structure",
                f"The selected folder is missing: {', '.join(missing_folders)}\n\n"
                f"Expected structure:\n{path}/\n  ├── train/\n  ├── valid/\n  └── test/"
            )
        else:
            self.log(f"✅ Valid dataset structure: {path}")

    def log(self, message):
        """Add message to log window"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def clear_log(self):
        """Clear the log window"""
        self.log_text.delete(1.0, tk.END)

    def start_training(self):
        """Start the training script in a separate thread"""
        dataset = self.dataset_path.get().strip()

        if not dataset:
            messagebox.showerror("Error", "Please select a dataset folder first!")
            return

        if not os.path.isdir(dataset):
            messagebox.showerror("Error", f"Dataset path does not exist:\n{dataset}")
            return

        # Disable start button, enable stop button
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.training_running = True

        # Clear previous log
        self.clear_log()
        self.log("=" * 60)
        self.log("  FEDERATED LEARNING - CLIENT TRAINING")
        self.log("=" * 60)
        self.log(f"  Client: {self.client_name.get()} ({self.client_id.get()})")
        self.log(f"  Dataset: {dataset}")
        self.log(f"  Epochs: {self.epochs.get()} | Batch: {self.batch_size.get()} | LR: {self.learning_rate.get()}")
        self.log("=" * 60 + "\n")

        self.status_var.set("Training in progress...")
        self.status_indicator.config(fg=DarkTheme.ACCENT_INFO)

        # Run training in separate thread
        training_thread = threading.Thread(target=self.run_training_script, daemon=True)
        training_thread.start()

    def run_training_script(self):
        """Execute the training and upload script"""
        dataset = self.dataset_path.get().strip()

        cmd = [
            "python",
            "fl_client_train_and_upload.py",
            "--dataset-path", dataset,
            "--epochs", str(self.epochs.get()),
            "--batch-size", str(self.batch_size.get()),
            "--learning-rate", str(self.learning_rate.get()),
            "--client-id", self.client_id.get(),
            "--client-name", self.client_name.get()
        ]

        self.log(f"▶ Running: {' '.join(cmd)}\n")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            for line in iter(self.process.stdout.readline, ''):
                if not self.training_running:
                    break
                self.log(line.rstrip())

            self.process.wait()

            if self.process.returncode == 0:
                self.log("\n" + "=" * 60)
                self.log("  ✅ TRAINING COMPLETED SUCCESSFULLY!")
                self.log("=" * 60)
                self.status_var.set("✅ Training completed")
                self.status_indicator.config(fg=DarkTheme.ACCENT_SUCCESS)
                messagebox.showinfo("Success", "Training completed!\nWeights uploaded to FL server.")
            else:
                self.log("\n" + "=" * 60)
                self.log(f"  ❌ Training failed (exit code: {self.process.returncode})")
                self.log("=" * 60)
                self.status_var.set(f"❌ Training failed")
                self.status_indicator.config(fg=DarkTheme.ACCENT_DANGER)
                messagebox.showerror("Error", f"Training failed.\nCheck the log for details.")

        except FileNotFoundError:
            self.log("❌ Error: Training script not found!")
            self.status_var.set("❌ Script not found")
            self.status_indicator.config(fg=DarkTheme.ACCENT_DANGER)
            messagebox.showerror("Error", "fl_client_train_and_upload.py not found!")

        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            self.status_var.set(f"❌ Error")
            self.status_indicator.config(fg=DarkTheme.ACCENT_DANGER)
            messagebox.showerror("Error", str(e))

        finally:
            self.training_running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.process = None

    def stop_training(self):
        """Stop the running training process"""
        if self.process and self.training_running:
            self.log("\n⚠️ Stopping training...")
            self.training_running = False
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

            self.log("🛑 Training stopped by user")
            self.status_var.set("🛑 Training stopped")
            self.status_indicator.config(fg=DarkTheme.ACCENT_WARNING)
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)


def main():
    """Main entry point"""
    root = tk.Tk()

    # Set window icon (if available)
    try:
        root.iconname("FL Client")
    except:
        pass

    app = FederatedLearningGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
