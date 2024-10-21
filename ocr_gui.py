from tkinter import scrolledtext
from turtle import mode
from win32 import win32api, win32gui, win32print
from win32.lib import win32con
from win32.win32api import GetSystemMetrics
import tkinter as tk
from PIL import ImageGrab, Image, ImageTk
import pyperclip
import latex2mathml.converter
import models
import utils

def get_real_resolution():
    hDC = win32gui.GetDC(0)
    # 横向分辨率
    w = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
    # 纵向分辨率
    h = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
    return w, h


import sys
import os


def get_screen_size():
    # 获取缩放后的分辨率
    w = GetSystemMetrics(0)
    h = GetSystemMetrics(1)
    return w, h


real_resolution = get_real_resolution()
screen_size = get_screen_size()
# Windows 设置的屏幕缩放率
# ImageGrab 的参数是基于显示分辨率的坐标，而 tkinter 获取到的是基于缩放后的分辨率的坐标
screen_scale_rate = round(real_resolution[0] / screen_size[0], 2)


class Box:

    def __init__(self):
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None

    def isNone(self):
        return self.start_x is None or self.end_x is None

    def setStart(self, x, y):
        self.start_x = x
        self.start_y = y

    def setEnd(self, x, y):
        self.end_x = x
        self.end_y = y

    def box(self):
        lt_x = min(self.start_x, self.end_x)
        lt_y = min(self.start_y, self.end_y)
        rb_x = max(self.start_x, self.end_x)
        rb_y = max(self.start_y, self.end_y)
        return lt_x, lt_y, rb_x, rb_y

    def center(self):
        center_x = (self.start_x + self.end_x) / 2
        center_y = (self.start_y + self.end_y) / 2
        return center_x, center_y


class SelectionArea:

    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.area_box = Box()

    def empty(self):
        return self.area_box.isNone()

    def setStartPoint(self, x, y):
        self.canvas.delete("area", "lt_txt", "rb_txt")
        self.area_box.setStart(x, y)
        self.canvas.create_text(x, y - 10, text=f"({x}, {y})", fill="red", tag="lt_txt")

    def updateEndPoint(self, x, y):
        self.area_box.setEnd(x, y)
        self.canvas.delete("area", "rb_txt")
        box_area = self.area_box.box()
        # 画矩形区域
        self.canvas.create_rectangle(
            *box_area, fill="black", outline="red", width=2, tags="area"
        )
        self.canvas.create_text(x, y + 10, text=f"({x}, {y})", fill="red", tag="rb_txt")


class ScreenShot:

    def __init__(self, scaling_factor=2, main=None):
        self.win = tk.Tk()
        # self.win.tk.call('tk', 'scaling', scaling_factor)
        self.width = self.win.winfo_screenwidth()
        self.height = self.win.winfo_screenheight()
        self.main = main
        # 无边框，无法拖动的半透明窗口
        self.win.overrideredirect(True)
        self.win.attributes("-alpha", 0.25)

        self.is_selecting = False

        # Enter 确认, Esc 退出
        self.win.bind("<KeyPress-Escape>", self.exit)
        self.win.bind("<KeyPress-Return>", self.confirmScreenShot)
        self.win.bind("<Button-1>", self.selectStart)
        self.win.bind("<ButtonRelease-1>", self.selectDone)
        self.win.bind("<Motion>", self.changeSelectionArea)

        self.canvas = tk.Canvas(self.win, width=self.width, height=self.height)
        self.canvas.pack()
        self.area = SelectionArea(self.canvas)
        self.win.mainloop()

    def exit(self, event):
        self.win.quit()
        self.win.destroy()

    def clear(self):
        self.canvas.delete("area", "lt_txt", "rb_txt")
        self.win.attributes("-alpha", 0)

    def captureImage(self):
        if self.area.empty():
            return None
        else:
            box_area = [x * screen_scale_rate for x in self.area.area_box.box()]
            self.clear()
            print(f"Grab: {box_area}")
            img = ImageGrab.grab(box_area)

            return img

    def confirmScreenShot(self, event):
        img = self.captureImage()
        self.main.image = img
        self.win.quit()
        self.win.destroy()

    def selectStart(self, event):
        self.is_selecting = True
        self.area.setStartPoint(event.x, event.y)

    def changeSelectionArea(self, event):
        if self.is_selecting:
            self.area.updateEndPoint(event.x, event.y)

    def selectDone(self, event):
        self.is_selecting = False


class MainWindow:
    def __init__(self, image_path=None, ocr=None):
        self.root = tk.Tk()
        # 窗口在最前
        self.root.attributes("-topmost",1)
        # 固定窗口尺寸
        if getattr(sys, "frozen", None):
            basedir = sys._MEIPASS
        else:
            basedir = os.path.dirname(__file__)
        self.root.iconbitmap(os.path.join(basedir, "models/guiicon.ico"))
        self.root.resizable(0, 0)
        self.ocr = ocr
        self.mathtypetext = ""
        self.latextext = ""
        if image_path is None:
            raise ValueError("Image path must be provided.")

        self.root.title("Free Formula OCR for Latex and MathML -~- By ltc")
        self.root.geometry("500x210")

        # 加载并显示图片
        self.image = Image.open(image_path)
        self.image = self.image.resize((500, 125))
        self.photo = ImageTk.PhotoImage(self.image)

        self.image_label = tk.Label(self.root, image=self.photo)
        self.image_label.image = self.photo  # 保持对图片的引用
        self.image_label.pack(pady=0)

        # 创建按钮框架和按钮
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(fill=tk.X, padx=45, pady=5)

        self.copy_latex_button = tk.Button(
            self.button_frame,
            text="Copy Latex",
            width=12,
            height=2,
            command=self.on_copy_latex,
        )
        self.copy_latex_button.grid(row=0, column=0, padx=5, pady=5)

        self.copy_mathtype_button = tk.Button(
            self.button_frame,
            text="Copy MathMl",
            width=12,
            height=2,
            command=self.on_copy_mathtype,
        )
        self.copy_mathtype_button.grid(row=0, column=1, padx=5, pady=5)

        self.snip_button = tk.Button(
            self.button_frame, text="Snipping", width=12, height=2, command=self.on_snip
        )
        self.snip_button.grid(row=0, column=2, padx=5, pady=5)

        self.help_button = tk.Button(
            self.button_frame, text="Help", width=12, height=2, command=self.on_help
        )
        self.help_button.grid(row=0, column=3, padx=5, pady=5)

        self.root.mainloop()

    def updatephoto(self):
        self.image = self.image.resize((500, 125))
        self.photo = ImageTk.PhotoImage(self.image)
        self.image_label.config(image=self.photo)

    def on_copy_latex(self):
        print("Copy LaTeX command executed.")
        pyperclip.copy(self.latextext)

    def on_copy_mathtype(self):
        print("Copy MathType command executed.")
        pyperclip.copy(self.mathtypetext)

    def on_snip(self):
        ScreenShot(main=self)
        print("Start cal")
        self.latextext = self.ocr.predict(self.image)["formula"]
        self.mathtypetext = utils.postprocessForMathml(
            latex2mathml.converter.convert(self.latextext)
        )
        self.updatephoto()
        print("Snip command executed.")



    def on_help(self):
        top = tk.Toplevel()
        top.title("Help")

        msg = tk.Message(
            top,
            width=410,
            text="点击snipping，鼠标左键选择截图区域，Enter键确定，Esc键退出。完成识别后，显示的图片会变化，此时即可以复制Latex或MathML格式的公式。\n满意的话请到Github给个赞，后续改进也会在上面发布：\n https://github.com/ai25395/FMatPix ",
        )
        msg.pack()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    if getattr(sys, "frozen", None):
        basedir = sys._MEIPASS
    else:
        basedir = os.path.dirname(__file__)
    model = models.Latex_OCR()
    MainWindow(os.path.join(basedir, "models/GuiStartImg.png"), model)
