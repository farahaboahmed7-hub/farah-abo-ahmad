import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# ==================== متغيرات المشروع ====================
current_img1 = None
current_img2 = None
merged_img = None
original_img1 = None
original_img2 = None

# ==================== دوال تحديث الصورة ====================

def update_image():
    global current_img1, current_img2, merged_img
    
    if merged_img is not None:
        img_pil = Image.fromarray(merged_img)
    elif current_img1 is not None:
        img_pil = Image.fromarray(current_img1)
    elif current_img2 is not None:
        img_pil = Image.fromarray(current_img2)
    else:
        return

    img_pil = img_pil.resize((500, 350))
    img_tk = ImageTk.PhotoImage(img_pil)
    label.config(image=img_tk)
    label.image = img_tk

def load_default_images():
    global current_img1, current_img2, original_img1, original_img2
    img1 = np.zeros((400, 600, 3), dtype=np.uint8)
    img1[:, :] = [100, 150, 200]
    cv2.rectangle(img1, (50, 50), (250, 350), (255, 0, 0), -1)
    cv2.putText(img1, "Image 1", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    img2 = np.zeros((400, 600, 3), dtype=np.uint8)
    img2[:, :] = [200, 150, 100]
    cv2.circle(img2, (500, 200), 80, (0, 255, 0), -1)
    cv2.putText(img2, "Image 2", (380, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    current_img1 = img1
    current_img2 = img2
    original_img1 = img1.copy()
    original_img2 = img2.copy()
    update_image()

# ==================== دوال الفلاتر ====================

def apply_to_active(func):
    """دالة مساعدة لتطبيق الفلتر على الصورة النشطة"""
    global current_img1, current_img2, merged_img
    if merged_img is not None:
        merged_img = func(merged_img)
    elif current_img1 is not None:
        current_img1 = func(current_img1)
    elif current_img2 is not None:
        current_img2 = func(current_img2)
    else:
        messagebox.showwarning("تنبيه", "الرجاء رفع صورة أولاً!")
        return
    update_image()

def gray_image():
    apply_to_active(lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB))

def blur_image():
    apply_to_active(lambda img: cv2.GaussianBlur(img, (15, 15), 0))

def edge_image():
    def _edge(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    apply_to_active(_edge)

def sharpen_image():
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    apply_to_active(lambda img: cv2.filter2D(img, -1, kernel))

def brighten_image():
    apply_to_active(lambda img: cv2.convertScaleAbs(img, alpha=1.2, beta=30))

def invert_image():
    apply_to_active(lambda img: cv2.bitwise_not(img))

def sepia_image():
    kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
    apply_to_active(lambda img: np.clip(cv2.transform(img, kernel), 0, 255).astype(np.uint8))

def rotate_image():
    apply_to_active(lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))

def darken_image():
    apply_to_active(lambda img: cv2.convertScaleAbs(img, alpha=0.8, beta=-20))

def cool_filter():
    def _cool(img):
        res = img.copy()
        res[:, :, 0] = cv2.multiply(res[:, :, 0], 0.8)
        res[:, :, 2] = cv2.multiply(res[:, :, 2], 1.2)
        return np.clip(res, 0, 255).astype(np.uint8)
    apply_to_active(_cool)

def warm_filter():
    def _warm(img):
        res = img.copy()
        res[:, :, 0] = cv2.multiply(res[:, :, 0], 1.2)
        res[:, :, 2] = cv2.multiply(res[:, :, 2], 0.8)
        return np.clip(res, 0, 255).astype(np.uint8)
    apply_to_active(_warm)

def cartoon_filter():
    apply_to_active(lambda img: cv2.stylization(img, sigma_s=60, sigma_r=0.4))

def neon_filter():
    apply_to_active(lambda img: np.clip(img.astype(np.float32) * 1.5, 0, 255).astype(np.uint8))

def pixelate_filter():
    def _pixel(img):
        h, w = img.shape[:2]
        temp = cv2.resize(img, (w // 10, h // 10))
        return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    apply_to_active(_pixel)

def thermal_filter():
    def _thermal(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
    apply_to_active(_thermal)

def posterize_filter():
    apply_to_active(lambda img: (img // 64) * 64)

def emboss_filter():
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    def _emboss(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        res = cv2.filter2D(gray, -1, kernel)
        return cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
    apply_to_active(_emboss)

def reset_image():
    global current_img1, current_img2, merged_img
    if original_img1 is not None: current_img1 = original_img1.copy()
    if original_img2 is not None: current_img2 = original_img2.copy()
    merged_img = None
    update_image()
    messagebox.showinfo("نجاح", "تمت الإعادة للأصل!")

def save_image():
    img_to_save = merged_img if merged_img is not None else (current_img1 if current_img1 is not None else current_img2)
    if img_to_save is None:
        messagebox.showwarning("تنبيه", "لا توجد صورة!")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".png")
    if file_path:
        cv2.imwrite(file_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
        messagebox.showinfo("نجاح", "تم الحفظ!")

# ==================== دوال الرفع والدمج ====================

def upload_image1():
    global current_img1, original_img1, merged_img
    path = filedialog.askopenfilename()
    if path:
        img = cv2.imread(path)
        if img is not None:
            current_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_img1 = current_img1.copy()
            merged_img = None
            update_image()

def upload_image2():
    global current_img2, original_img2, merged_img
    path = filedialog.askopenfilename()
    if path:
        img = cv2.imread(path)
        if img is not None:
            current_img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_img2 = current_img2.copy()
            merged_img = None
            update_image()

def merge_images():
    global current_img1, current_img2, merged_img
    if current_img1 is None or current_img2 is None:
        messagebox.showwarning("تنبيه", "ارفع الصورتين أولاً")
        return
    h, w = current_img1.shape[:2]
    img2_resized = cv2.resize(current_img2, (w, h))
    merged_img = cv2.addWeighted(current_img1, 0.5, img2_resized, 0.5, 0)
    update_image()

# ==================== الواجهة الرسومية ====================

root = tk.Tk()
root.title("محرر دمج الصور الذكي")
root.geometry("1200x850")
root.configure(bg="#0f0f0f")

# العنوان
Label(root, text="🎨 IMAGE MERGER PRO", font=("Impact", 32), bg="#0f0f0f", fg="#00ff88").pack(pady=10)

# منطقة العرض الكبيرة
main_image_frame = Frame(root, bg="#1a1a1a", bd=2, relief="flat")
main_image_frame.pack(expand=True, fill="both", padx=50, pady=10)

label = Label(main_image_frame, bg="#1a1a1a")
label.pack(expand=True)

# --- منطقة الأزرار المنظمة ---
bottom_frame = Frame(root, bg="#0f0f0f")
bottom_frame.pack(fill="x", side="bottom", padx=20, pady=20)

# ستايل الأزرار
btn_style = {"font": ("Arial", 9, "bold"), "width": 12, "pady": 5, "cursor": "hand2", "relief": "flat"}

# 1. مجموعة الرفع والدمج
grp1 = LabelFrame(bottom_frame, text=" الصور والدمج ", bg="#0f0f0f", fg="#00ff88", font=("Arial", 10, "bold"), padx=10, pady=10)
grp1.pack(side="left", padx=5, fill="y")
Button(grp1, text="رفع صورة 1", command=upload_image1, bg="#00ff88", fg="#000", **btn_style).grid(row=0, column=0, pady=2)
Button(grp1, text="رفع صورة 2", command=upload_image2, bg="#00ff88", fg="#000", **btn_style).grid(row=1, column=0, pady=2)
Button(grp1, text="دمج الآن", command=merge_images, bg="#ff6b9d", fg="#fff", **btn_style).grid(row=2, column=0, pady=5)

# 2. مجموعة الألوان
grp2 = LabelFrame(bottom_frame, text=" فلاتر الألوان ", bg="#0f0f0f", fg="#9370db", font=("Arial", 10, "bold"), padx=10, pady=10)
grp2.pack(side="left", padx=5, fill="y")
Button(grp2, text="رمادي", command=gray_image, bg="#9370db", fg="#fff", **btn_style).grid(row=0, column=0, padx=2, pady=2)
Button(grp2, text="سيبيا", command=sepia_image, bg="#8B7355", fg="#fff", **btn_style).grid(row=0, column=1, padx=2, pady=2)
Button(grp2, text="عكس الألوان", command=invert_image, bg="#696969", fg="#fff", **btn_style).grid(row=1, column=0, padx=2, pady=2)
Button(grp2, text="فلتر بارد", command=cool_filter, bg="#0099ff", fg="#fff", **btn_style).grid(row=1, column=1, padx=2, pady=2)
Button(grp2, text="فلتر دافئ", command=warm_filter, bg="#ff6633", fg="#fff", **btn_style).grid(row=2, column=0, padx=2, pady=2)
Button(grp2, text="بوستر", command=posterize_filter, bg="#cc66ff", fg="#fff", **btn_style).grid(row=2, column=1, padx=2, pady=2)

# 3. مجموعة التأثيرات
grp3 = LabelFrame(bottom_frame, text=" تأثيرات تقنية ", bg="#0f0f0f", fg="#ffa500", font=("Arial", 10, "bold"), padx=10, pady=10)
grp3.pack(side="left", padx=5, fill="y")
Button(grp3, text="تمويه", command=blur_image, bg="#1e90ff", fg="#fff", **btn_style).grid(row=0, column=0, padx=2, pady=2)
Button(grp3, text="حواف", command=edge_image, bg="#ff6347", fg="#fff", **btn_style).grid(row=0, column=1, padx=2, pady=2)
Button(grp3, text="حدة", command=sharpen_image, bg="#ffa500", fg="#fff", **btn_style).grid(row=1, column=0, padx=2, pady=2)
Button(grp3, text="إضاءة", command=brighten_image, bg="#ffbf00", fg="#000", **btn_style).grid(row=1, column=1, padx=2, pady=2)
Button(grp3, text="تدوير", command=rotate_image, bg="#ff1493", fg="#fff", **btn_style).grid(row=2, column=0, padx=2, pady=2)
Button(grp3, text="عتمة", command=darken_image, bg="#333", fg="#fff", **btn_style).grid(row=2, column=1, padx=2, pady=2)

# 4. مجموعة الفلاتر الفنية (بدون زر الرسم الزيتي)
grp4 = LabelFrame(bottom_frame, text=" فلاتر فنية ", bg="#0f0f0f", fg="#daa520", font=("Arial", 10, "bold"), padx=10, pady=10)
grp4.pack(side="left", padx=5, fill="y")
Button(grp4, text="كرتون", command=cartoon_filter, bg="#ff00ff", fg="#fff", **btn_style).grid(row=0, column=0, padx=2, pady=2)
Button(grp4, text="نيون", command=neon_filter, bg="#00ff00", fg="#000", **btn_style).grid(row=0, column=1, padx=2, pady=2)
Button(grp4, text="بيكسل", command=pixelate_filter, bg="#4b0082", fg="#fff", **btn_style).grid(row=1, column=0, padx=2, pady=2)
Button(grp4, text="حراري", command=thermal_filter, bg="#ff4500", fg="#fff", **btn_style).grid(row=1, column=1, padx=2, pady=2)
Button(grp4, text="إغاثة", command=emboss_filter, bg="#daa520", fg="#fff", **btn_style).grid(row=2, column=0, padx=2, pady=2)

# 5. الحفظ والإعدادات
grp5 = LabelFrame(bottom_frame, text=" الأدوات ", bg="#0f0f0f", fg="#fff", font=("Arial", 10, "bold"), padx=10, pady=10)
grp5.pack(side="left", padx=5, fill="y")
Button(grp5, text="إعادة تعيين", command=reset_image, bg="#228b22", fg="#fff", **btn_style).grid(row=0, column=0, pady=5)
Button(grp5, text="حفظ الصورة", command=save_image, bg="#dc143c", fg="#fff", **btn_style).grid(row=1, column=0, pady=5)

# تشغيل الصور الافتراضية
root.after(200, load_default_images)
root.mainloop()
