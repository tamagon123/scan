import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog
import os

class PCVFlatApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("PC vFlat - Document Scanner")
        self.geometry("1000x700")

        # データの初期化
        self.raw_image = None
        self.processed_image = None

        # UIのレイアウト
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 左側メニュー
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="PC vFlat Tool", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=20)

        self.btn_open = ctk.CTkButton(self.sidebar, text="画像を開く", command=self.load_image)
        self.btn_open.pack(pady=10, padx=20)

        self.btn_process = ctk.CTkButton(self.sidebar, text="スキャン加工実行", command=self.process_scan, state="disabled")
        self.btn_process.pack(pady=10, padx=20)

        self.btn_save = ctk.CTkButton(self.sidebar, text="保存する", command=self.save_image, state="disabled", fg_color="green")
        self.btn_save.pack(pady=10, padx=20)

        # 中央プレビューエリア
        self.preview_frame = ctk.CTkFrame(self)
        self.preview_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.image_label = ctk.CTkLabel(self.preview_frame, text="画像を選択してください")
        self.image_label.pack(expand=True, fill="both")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if path:
            self.raw_image = cv2.imread(path)
            self.show_image(self.raw_image)
            self.btn_process.configure(state="normal")

    def show_image(self, cv_img):
        # OpenCVのBGRをRGBに変換して表示用にリサイズ
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # プレビューサイズに合わせてリサイズ
        display_size = (800, 600)
        img_pil.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_label.configure(image=img_tk, text="")
        self.image_label.image = img_tk

    def process_scan(self):
        if self.raw_image is None: return

        # 1. グレースケール化とノイズ除去
        img = self.raw_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. エッジ検出
        edged = cv2.Canny(blurred, 75, 200)

        # 3. 輪郭抽出
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        screen_cnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4: # 四角形が見つかったら
                screen_cnt = approx
                break

        if screen_cnt is not None:
            # 四角形が見つかった場合：透視変換（フラット化）
            pts = screen_cnt.reshape(4, 2)
            rect = self.order_points(pts)
            (tl, tr, br, bl) = rect

            width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            max_width = max(int(width_a), int(width_b))

            height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_height = max(int(height_a), int(height_b))

            dst = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (max_width, max_height))
            self.processed_image = warped
        else:
            # 四角形が見つからない場合：そのまま続行
            self.processed_image = img

        # 4. vFlat風の「スキャン仕上げ」加工
        self.processed_image = self.apply_scan_effect(self.processed_image)
        
        self.show_image(self.processed_image)
        self.btn_save.configure(state="normal")

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def apply_scan_effect(self, img):
        # グレースケール化して適応的二値化（白背景を強調）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scan = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
        # カラー感を少し残す場合は混合も可能ですが、まずは読みやすさ重視で2値化
        return cv2.cvtColor(scan, cv2.COLOR_GRAY2BGR)

    def save_image(self):
        if self.processed_image is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg")])
            if save_path:
                cv2.imwrite(save_path, self.processed_image)

if __name__ == "__main__":
    app = PCVFlatApp()
    app.mainloop()