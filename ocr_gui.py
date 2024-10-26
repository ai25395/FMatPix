from ctypes import util
import queue
from transformers.modeling_utils import safe_load_file
from win32 import win32api, win32gui, win32print
from win32.lib import win32con
from win32.win32api import GetSystemMetrics
import tkinter as tk
from PIL import ImageGrab, Image
import pyperclip
import models
import utils
from pynput import keyboard
import sys
import os
import threading
import queue
import time
def get_real_resolution():
    hDC = win32gui.GetDC(0)
    # 横向分辨率
    w = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
    # 纵向分辨率
    h = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
    return w, h
def get_screen_size():
    # 获取缩放后的分辨率
    w = GetSystemMetrics(0)
    h = GetSystemMetrics(1)
    return w, h
# Windows 设置的屏幕缩放率
# ImageGrab 的参数是基于显示分辨率的坐标，而 tkinter 获取到的是基于缩放后的分辨率的坐标
real_resolution = get_real_resolution()
screen_size = get_screen_size()
screen_scale_rate = round(real_resolution[0] / screen_size[0], 2)
capturedImg = None



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

#截图区域控件
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
        self.canvas.create_rectangle(
            *box_area, fill="black", outline="red", width=2, tags="area"
        )
        self.canvas.create_text(x, y + 10, text=f"({x}, {y})", fill="red", tag="rb_txt")



#截图总界面控件
class ScreenShot:
    def __init__(self, scaling_factor=2, main=None):
        self.win = tk.Tk()
        self.width = self.win.winfo_screenwidth()
        self.height = self.win.winfo_screenheight()
        #self.main = main
        self.win.overrideredirect(True)
        self.win.attributes("-alpha", 0.25)
        self.is_selecting = False
        self.win.bind("<KeyPress-Escape>", self.exit)
        self.win.bind("<Button-1>", self.selectStart)
        self.win.bind("<ButtonRelease-1>", self.selectDone)
        self.win.bind("<Motion>", self.changeSelectionArea)
        self.canvas = tk.Canvas(self.win, width=self.width, height=self.height)
        self.canvas.pack()
        self.area = SelectionArea(self.canvas)
        self.win.attributes("-topmost",1)
        self.win.focus_set()
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
            global capturedImg
            capturedImg = ImageGrab.grab(box_area)
            #return img

    def selectStart(self, event):
        self.is_selecting = True
        self.area.setStartPoint(event.x, event.y)

    def changeSelectionArea(self, event):
        if self.is_selecting:
            self.area.updateEndPoint(event.x, event.y)

    def selectDone(self, event):
        self.is_selecting = False
        self.captureImage()
        self.exit(None)


#FMatPix界面
class MainWindow:
    def __init__(self, image_path=None, ocr=None, detector=None):
        self.root = tk.Tk()
        #默认窗口最前，如果最小化，则取消窗口最前；最大化，则恢复窗口最前。
        self.root.attributes("-topmost",1)
        self.root.bind('<Map>',lambda x:self.root.attributes("-topmost",1))
        self.root.bind('<Unmap>',lambda x:self.root.attributes("-topmost",0))
        # 固定窗口尺寸
        self.root.resizable(0, 0)
        if getattr(sys, "frozen", None):
            basedir = sys._MEIPASS
        else:
            basedir = os.path.dirname(__file__)
        self.root.iconbitmap(os.path.join(basedir, "models/guiicon.ico"))
        self.timeout = 60
        self.ocr = ocr
        self.detector = detector
        self.mathtypetext = ""
        self.latextext = ""
        self.history = [[],[]]
        self.autocopylatex = False
        self.autocopymathml = False
        if image_path is None:
            raise ValueError("Image path must be provided.")

        self.root.title("Free Formula OCR for Latex and MathML v1.3 -~- Ltc")
        self.root.geometry("500x70")
        
        #控制同时只能有一个Screen
        self.hasScreen = False
        #初次运行读取预设图片
        if getattr(sys, "frozen", None):
            basedir = sys._MEIPASS
        else:
            basedir = os.path.dirname(__file__)
        self.image = Image.open(os.path.join(basedir,"models/GuiStartImg.PNG"))
        #self.runCalWithTimeout(self.cal,self.image,8)
        # 创建按钮框架和按钮
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(fill=tk.X, padx=20, pady=5)
        self.progress_label = tk.Label(self.button_frame, bg='green',fg='white', width=2, height=1)
        self.progress_label.grid(row=0, column=0, padx=5, pady=5)
        self.copy_latex_button = tk.Button(
            self.button_frame,
            text="Copy Latex",
            width=12,
            height=2,
            command=self.on_copy_latex,
        )
        self.copy_latex_button.bind("<Double-Button-1>", self.on_doubleclick_latex)
        self.copy_latex_button.grid(row=0, column=1, padx=5, pady=5)
        self.copy_mathtype_button = tk.Button(
            self.button_frame,
            text="Copy MathMl",
            width=12,
            height=2,
            command=self.on_copy_mathtype,
        )
        self.copy_mathtype_button.bind("<Double-Button-1>", self.on_doubleclick_mathml)
        self.copy_mathtype_button.grid(row=0, column=2, padx=5, pady=5)
        self.history_button = tk.Button(
            self.button_frame, text="History", width=12, height=2, command=self.copy_history_copy
        )
        self.history_button.bind("<Double-Button-1>", self.on_history_refresh)
        self.history_button.grid(row=0, column=3, padx=5, pady=5)
        self.help_button = tk.Button(
            self.button_frame, text="Help", width=12, height=2, command=self.on_help
        )
        self.help_button.grid(row=0, column=4, padx=5, pady=5)
    
    #复制latex
    def on_copy_latex(self):
        print("Copy LaTeX command executed.")
        pyperclip.copy(self.latextext)
    
    #开启或关闭latex自动复制
    def on_doubleclick_latex(self,event):
        if self.autocopylatex == True:
            self.autocopylatex = False
            self.copy_latex_button.config(bg='SystemButtonFace')
            return 

        self.copy_latex_button.config(bg='DarkGray')
        self.autocopylatex = True
        if self.autocopymathml == True:
            self.autocopymathml = False
            self.copy_mathtype_button.config(bg='SystemButtonFace')
    #开启或关闭mathml自动复制
    def on_doubleclick_mathml(self,event):
        if self.autocopymathml == True:
            self.autocopymathml = False
            self.copy_mathtype_button.config(bg='SystemButtonFace')
            return
        self.copy_mathtype_button.config(bg='DarkGray')
        self.autocopymathml = True
        if self.autocopylatex == True:
            self.autocopylatex = False
            self.copy_latex_button.config(bg='SystemButtonFace')
    #复制mathml
    def on_copy_mathtype(self):
        print("Copy MathType command executed.")
        pyperclip.copy(self.mathtypetext)
    
    #复制历史公式，同时有Latex和Mathml
    def copy_history_copy(self):
        #复制self.results到clipboard
        if self.history==[[],[]]:
            pyperclip.copy('no history')
            return
        out = ''
        for res in self.history[0]:
            out += res+'\n'
        out += '\n\n\n'
        for res in self.history[1]:
            out += res+'\n'
        pyperclip.copy(out)
        
    #清空历史公式
    def on_history_refresh(self,event):
        print('清空历史公式')
        self.history = [[],[]]
            
    #核心识别函数
    def cal(self,img):
        try:
            print('识别')
            imgs,alignment = self.detector.detect_image(img)
            print('检测公式数量')
            print(len(imgs))
            results = self.ocr.predict(imgs)
            latextext = utils.gatherOcrResults(results,alignment)
            print('转换')
            tmp1 = utils.InvokeTexmml(latextext)
            print('后处理')
            mathtypetext = utils.postprocessForMathml(tmp1)
            return (latextext,mathtypetext)
        except Exception as e:
            print("核心识别函数出错")
            print(e)
            return ('','')
    #调用cal，并在超时时结束
    def runCalWithTimeout(self,func,img,timeout):
        q = queue.Queue()
        def wrapper(q):
            try:
                res = func(img)
                q.put(res)
            except Exception as e:
                print(f"Function raised an exception: {e}")
                pass
        thread = threading.Thread(target=wrapper,args=(q,))
        #thread.setDaemon(True)
        thread.start()
        thread.join(timeout)
        # 如果线程仍然存活，说明超时了
        if thread.is_alive():
            print("Function timed out.")
            raise Exception
        #更新latextext和mathtypetext
        try:
            res = q.get_nowait() 
            self.latextext = res[0]
            self.mathtypetext = res[1]
            self.history[0].append(self.latextext)
            self.history[1].append(self.mathtypetext)
            if self.autocopylatex:
                self.on_copy_latex()
            if self.autocopymathml:
                self.on_copy_mathtype()
        except queue.Empty as e:
            print("识别结果为空")
            pass
        
    #打开截图窗口并识别
    def on_snip(self,snip=None):
        #禁止同时出现两个截图窗口
        if self.hasScreen==True:
            return
        self.hasScreen = True
        
        print("执行开始")
        try:
            print('截图')
            ScreenShot(main=self)
            #self.image = capturedImg
            print('状态灯更新为红')
            self.progress_label.config(background='red')
            self.root.update()
            print('开始计算')
            global capturedImg
            if capturedImg:
                t1 = time.time()
                self.runCalWithTimeout(self.cal,capturedImg,self.timeout)
                print('检测+识别共用时：'+ str(time.time()-t1)+ 's')
            else:
                print('image==None')
            self.hasScreen = False
            print('执行完成')
        except Exception as e:
            print(e)
            print('recognition error')
        finally:
            self.hasScreen = False
            self.progress_label.config(background='green')
            self.root.update()
    
    def on_help(self):
        top = tk.Toplevel()
        top.title("Help")
        top.attributes("-topmost",1)
        msg = tk.Message(
            top,
            width=450,
            text="1、按F2，鼠标左键选择截图区域，Esc键退出。\n 2、完成识别后，可单击copy latex/mathml来复制公式。\n 3、双击copy latex/mathml来开启或关闭自动复制。\n 4、单击History复制全部历史公式，双击则清空历史公式。 \n 5、红色状态灯表示正在识别，绿色状态灯表示识别完成。 \n 5、满意的话请到Github给个赞，后续改进也会在上面发布：\n https://github.com/ai25395/FMatPix ",
        )
        msg.pack()

    def run(self):
        self.root.mainloop()



if __name__ == "__main__":
    if getattr(sys, "frozen", None):
        basedir = sys._MEIPASS
    else:
        basedir = os.path.dirname(__file__)
    detector = utils.PreProcess(os.path.join(basedir, "models/best.onnx"))
    model = models.OcrModel()
    window = MainWindow(os.path.join(basedir, "models/GuiStartImg.png"), model, detector)
    #消息队列，用于键盘监听线程和主程序通信
    message_queue = queue.Queue()

    #键盘监听截图快捷键相关函数
    def on_press(key):
        if window.hasScreen:
            return
        try:
            if key == keyboard.Key.f2:
                message_queue.put('snip')
        except AttributeError:
            pass 
    def start_keyboard_listener():
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()    
    def checkqueue():
        try:
            message = message_queue.get_nowait()
            if message == 'snip':
                print('message')
                window.on_snip()
        except queue.Empty:
            pass
        window.root.after(200, checkqueue)
    
    #启动键盘监听线程
    listener_process = threading.Thread(target=start_keyboard_listener,daemon=True)
    listener_process.start()
    window.root.after(200, checkqueue)
    #启动主程序
    window.root.mainloop()
        
