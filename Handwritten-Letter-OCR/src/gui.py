import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import joblib
import os
from src.feature_extraction import FeatureExtractor
from scipy.ndimage import interpolation


def deskew(img):
    """Deskew an image using image moments (EMNIST style)."""
    c0, c1 = np.mgrid[:img.shape[0], :img.shape[1]]
    total_img = img.sum()
    if total_img == 0:
        return img
    m0 = (c0 * img).sum() / total_img
    m1 = (c1 * img).sum() / total_img
    mu11 = ((c0 - m0) * (c1 - m1) * img).sum() / total_img
    mu02 = ((c1 - m1) ** 2 * img).sum() / total_img
    if mu02 == 0:
        return img
    skew = mu11 / mu02
    return interpolation.affine_transform(img, [[1, 0], [-skew, 1]], offset=0, order=1, mode='constant', cval=0.0)


def softmax(x):
    """Softmax to scale Decision Tree probabilities."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class OCRGUI:
    def __init__(self, root, dt_model=None, rf_model=None, feature_extractor=None):
        self.root = root
        self.root.title("Handwritten Letter OCR")
        self.root.geometry("600x600")

        # Load models
        self.dt_model = dt_model if dt_model else self.load_model("decision_tree_model.pkl")
        self.rf_model = rf_model if rf_model else self.load_model("random_forest_model.pkl")
        self.feature_extractor = feature_extractor

        self.canvas_size = 280
        self.setup_ui()

        # Drawing canvas buffer
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(main_frame, text="Handwritten Letter OCR", font=("Arial", 16, "bold"))
        title.pack(pady=8)

        self.canvas = tk.Canvas(
            main_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='white',
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.canvas.pack(pady=8)

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=6)
        ttk.Button(btn_frame, text="Recognize from Canvas", command=self.recognize_letter).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Clear Canvas", command=self.clear_canvas).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Load Image", command=self.load_image_file).pack(side=tk.LEFT, padx=6)

        res_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
        res_frame.pack(fill=tk.BOTH, expand=True, pady=8)

        self.dt_label = ttk.Label(res_frame, text="Decision Tree:", font=("Arial", 12))
        self.dt_label.pack()
        self.rf_label = ttk.Label(res_frame, text="Random Forest:", font=("Arial", 12))
        self.rf_label.pack()

    # ---------- DRAWING ----------
    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=4,
                capstyle=tk.ROUND,
                smooth=True,
                splinesteps=36,
                fill="black"
            )
            self.draw.line((self.last_x, self.last_y, x, y), fill=0, width=4)
        self.last_x, self.last_y = x, y

    def reset(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.dt_label.config(text="Decision Tree:")
        self.rf_label.config(text="Random Forest:")

    # ---------- IMAGE PREPROCESS ----------
    def preprocess_image(self, img_arr):
        """Input: 2D numpy array. Returns flattened normalized 28x28."""
        # Invert for EMNIST style
        img_arr = 255 - img_arr

        # Auto-crop
        rows = np.where(img_arr.max(axis=1) > 10)[0]
        cols = np.where(img_arr.max(axis=0) > 10)[0]
        if rows.size and cols.size:
            img_arr = img_arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

        # Pad
        pad_y = img_arr.shape[0] // 8
        pad_x = img_arr.shape[1] // 8
        img_arr = np.pad(img_arr, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")

        # Resize to 28x28
        img28 = Image.fromarray(img_arr).resize((28, 28), Image.LANCZOS)
        img_arr = np.array(img28).astype(np.float32) / 255.0

        # Deskew
        img_arr = deskew(img_arr)
        return img_arr.flatten().reshape(1, -1)

    # ---------- PREDICTION ----------
    def recognize_letter(self):
        try:
            processed = np.array(self.image)
            features = self.feature_extractor.extract_features_batch(self.preprocess_image(processed))

            self.display_prediction(features)

        except Exception as e:
            messagebox.showerror("Error", f"Recognition failed:\n{e}")

    def display_prediction(self, features):
        # Decision Tree
        if self.dt_model:
            raw_dt = self.dt_model.predict_proba(features)[0]
            dt_proba = softmax(raw_dt)
            dt_pred = int(np.argmax(dt_proba))
            dt_letter = chr(65 + dt_pred)
            self.dt_label.config(text=f"Decision Tree → {dt_letter}  (Conf: {dt_proba[dt_pred]:.2%})")

        # Random Forest
        if self.rf_model:
            rf_proba = self.rf_model.predict_proba(features)[0]
            rf_pred = int(np.argmax(rf_proba))
            rf_letter = chr(65 + rf_pred)
            self.rf_label.config(text=f"Random Forest → {rf_letter}  (Conf: {rf_proba[rf_pred]:.2%})")

    # ---------- LOAD IMAGE FROM FILE ----------
    def load_image_file(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not file_path:
            return

        try:
            img = Image.open(file_path).convert("L")
            img_arr = np.array(img)
            features = self.feature_extractor.extract_features_batch(self.preprocess_image(img_arr))
            self.display_prediction(features)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or process image:\n{e}")

    def load_model(self, filename):
        path = os.path.join("models", filename)
        if os.path.exists(path):
            print(f"Loaded model: {filename}")
            return joblib.load(path)
        print(f"Model NOT found: {filename}")
        return None


if __name__ == "__main__":
    root = tk.Tk()
    # Load models and feature extractor from trained files
    fe = joblib.load("after.train.model/feature_extractor.pkl")
    dt = joblib.load("after.train.model/decision_tree_model.pkl")
    rf = joblib.load("after.train.model/random_forest_model.pkl")

    app = OCRGUI(root, dt_model=dt, rf_model=rf, feature_extractor=fe)
    root.mainloop()
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import joblib
import os
from src.feature_extraction import FeatureExtractor
from scipy.ndimage import interpolation


def deskew(img):
    """Deskew an image using image moments (EMNIST style)."""
    c0, c1 = np.mgrid[:img.shape[0], :img.shape[1]]
    total_img = img.sum()
    if total_img == 0:
        return img
    m0 = (c0 * img).sum() / total_img
    m1 = (c1 * img).sum() / total_img
    mu11 = ((c0 - m0) * (c1 - m1) * img).sum() / total_img
    mu02 = ((c1 - m1) ** 2 * img).sum() / total_img
    if mu02 == 0:
        return img
    skew = mu11 / mu02
    return interpolation.affine_transform(img, [[1, 0], [-skew, 1]], offset=0, order=1, mode='constant', cval=0.0)


def softmax(x):
    """Softmax to scale Decision Tree probabilities."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class OCRGUI:
    def __init__(self, root, dt_model=None, rf_model=None, feature_extractor=None):
        self.root = root
        self.root.title("Handwritten Letter OCR")
        self.root.geometry("600x600")

        # Load models
        self.dt_model = dt_model if dt_model else self.load_model("decision_tree_model.pkl")
        self.rf_model = rf_model if rf_model else self.load_model("random_forest_model.pkl")
        self.feature_extractor = feature_extractor

        self.canvas_size = 280
        self.setup_ui()

        # Drawing canvas buffer
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(main_frame, text="Handwritten Letter OCR", font=("Arial", 16, "bold"))
        title.pack(pady=8)

        self.canvas = tk.Canvas(
            main_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='white',
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.canvas.pack(pady=8)

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=6)
        ttk.Button(btn_frame, text="Recognize from Canvas", command=self.recognize_letter).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Clear Canvas", command=self.clear_canvas).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Load Image", command=self.load_image_file).pack(side=tk.LEFT, padx=6)

        res_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
        res_frame.pack(fill=tk.BOTH, expand=True, pady=8)

        self.dt_label = ttk.Label(res_frame, text="Decision Tree:", font=("Arial", 12))
        self.dt_label.pack()
        self.rf_label = ttk.Label(res_frame, text="Random Forest:", font=("Arial", 12))
        self.rf_label.pack()

    # ---------- DRAWING ----------
    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=4,
                capstyle=tk.ROUND,
                smooth=True,
                splinesteps=36,
                fill="black"
            )
            self.draw.line((self.last_x, self.last_y, x, y), fill=0, width=4)
        self.last_x, self.last_y = x, y

    def reset(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.dt_label.config(text="Decision Tree:")
        self.rf_label.config(text="Random Forest:")

    # ---------- IMAGE PREPROCESS ----------
    def preprocess_image(self, img_arr):
        """Input: 2D numpy array. Returns flattened normalized 28x28."""
        # Invert for EMNIST style
        img_arr = 255 - img_arr

        # Auto-crop
        rows = np.where(img_arr.max(axis=1) > 10)[0]
        cols = np.where(img_arr.max(axis=0) > 10)[0]
        if rows.size and cols.size:
            img_arr = img_arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

        # Pad
        pad_y = img_arr.shape[0] // 8
        pad_x = img_arr.shape[1] // 8
        img_arr = np.pad(img_arr, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")

        # Resize to 28x28
        img28 = Image.fromarray(img_arr).resize((28, 28), Image.LANCZOS)
        img_arr = np.array(img28).astype(np.float32) / 255.0
        # --- FIX ROTATION ---
        img_arr = np.rot90(img_arr, -1)
        img_arr = np.fliplr(img_arr)
        # Deskew
        img_arr = deskew(img_arr)
        return img_arr.flatten().reshape(1, -1)

    # ---------- PREDICTION ----------
    def recognize_letter(self):
        try:
            processed = np.array(self.image)
            features = self.feature_extractor.extract_features_batch(self.preprocess_image(processed))

            self.display_prediction(features)

        except Exception as e:
            messagebox.showerror("Error", f"Recognition failed:\n{e}")

    def display_prediction(self, features):

        # -------- Decision Tree --------
       if self.dt_model:
        dt_proba = self.dt_model.predict_proba(features)[0]
        dt_pred = int(np.argmax(dt_proba))
        dt_letter = chr(65 + dt_pred)

        self.dt_label.config(
            text=f"Decision Tree → {dt_letter}  (Conf: {dt_proba[dt_pred]:.2%})"
        )

    # -------- Random Forest --------
       if self.rf_model:
        rf_proba = self.rf_model.predict_proba(features)[0]
        rf_pred = int(np.argmax(rf_proba))
        rf_letter = chr(65 + rf_pred)

        self.rf_label.config(
            text=f"Random Forest → {rf_letter}  (Conf: {rf_proba[rf_pred]:.2%})"
        )


    # ---------- LOAD IMAGE FROM FILE ----------
    def load_image_file(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not file_path:
            return

        try:
            img = Image.open(file_path).convert("L")
            img_arr = np.array(img)
            features = self.feature_extractor.extract_features_batch(self.preprocess_image(img_arr))
            self.display_prediction(features)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or process image:\n{e}")

    def load_model(self, filename):
        path = os.path.join("models", filename)
        if os.path.exists(path):
            print(f"Loaded model: {filename}")
            return joblib.load(path)
        print(f"Model NOT found: {filename}")
        return None


if __name__ == "__main__":
    root = tk.Tk()
    # Load models and feature extractor from trained files
    fe = joblib.load("after.train.model/feature_extractor.pkl")
    dt = joblib.load("after.train.model/decision_tree_model.pkl")
    rf = joblib.load("after.train.model/random_forest_model.pkl")

    app = OCRGUI(root, dt_model=dt, rf_model=rf, feature_extractor=fe)
    root.mainloop()
