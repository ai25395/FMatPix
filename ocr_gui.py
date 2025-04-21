import queue
import tkinter as tk
from PIL import Image, ImageTk
import pyperclip
import models
import utils
from pynput import keyboard
import pynput
import sys
import os
import threading
import time
import mss
import ctypes


capturedImgs = []

# 定义所需的常量和函数
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

# 常量定义
DESKTOPHORZRES = 118  # Windows API 常量值，表示水平分辨率
DESKTOPVERTRES = 117  # Windows API 常量值，表示垂直分辨率
SM_CXSCREEN = 0       # 屏幕宽度指标
SM_CYSCREEN = 1       # 屏幕高度指标


def get_real_resolution():
    hdc = user32.GetDC(0)
    width = gdi32.GetDeviceCaps(hdc, DESKTOPHORZRES)
    height = gdi32.GetDeviceCaps(hdc, DESKTOPVERTRES)
    user32.ReleaseDC(0, hdc)
    return width, height

def get_screen_size():
    width = user32.GetSystemMetrics(SM_CXSCREEN)
    height = user32.GetSystemMetrics(SM_CYSCREEN)
    return width, height

real_resolution = get_real_resolution()
screen_size = get_screen_size()
screen_scale_rate = round(real_resolution[0] / screen_size[0], 2)

try:  # >= win 8.1
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except:  # win 8.0 or less
    ctypes.windll.user32.SetProcessDPIAware()

if sys.platform == "win32":
    try:
        dpi = user32.GetDpiForSystem()       
        sf = dpi / 96.0  # 96 DPI is the default for Windows
    except:
        sf = 1.0
else:
    sf = 1.0  # For non-Windows systems

#记录截图框坐标
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
        return [lt_x, lt_y, rb_x, rb_y]

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
        self.canvas.delete("area")
        self.area_box.setStart(x, y)

    def updateEndPoint(self, x, y,startW,startH):
        self.area_box.setEnd(x, y)
        self.canvas.delete("area")
        box_area = self.area_box.box()
        box_area[0] -= startW
        box_area[2] -= startW
        box_area[1] -= startH
        box_area[3] -= startH
        self.canvas.create_rectangle(
            *box_area, fill="black", outline="red", width=2, tags="area"
        )


#截图总界面控件
class ScreenShot:
    def __init__(self):
        self.win = tk.Tk()
        self.win.tk.call('tk', 'scaling', sf)
        self.sct = mss.mss()
        self.mouse = pynput.mouse.Controller()
        monitor0 = self.sct.monitors[0]
        self.startW,self.startH,self.width,self.height = monitor0['left'],monitor0['top'],monitor0['width'],monitor0['height']
        self.win.geometry(f"{self.width*2}x{self.height*2}+{self.startW}+{self.startH}")
        self.win.overrideredirect(True)
        self.win.attributes("-alpha", 0.25)
        self.is_selecting = False
        self.win.bind("<KeyPress-Escape>", self.exit)
        self.win.bind("<Button-1>", self.selectStart)
        self.win.bind("<ButtonRelease-1>", self.selectDone)
        self.win.bind("<Motion>", self.changeSelectionArea)
        self.canvas = tk.Canvas(self.win, width=self.width*2, height=self.height*2)
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
            box_area = [int(x * 1) for x in self.area.area_box.box()]
            region = {'left':box_area[0],'top':box_area[1],'width':box_area[2]-box_area[0],'height':box_area[3]-box_area[1]}
            self.clear()
            print(f"Grab: {box_area}")
            global capturedImgs
            img = self.sct.grab(region)
            
            capturedImgs.append(Image.frombytes('RGB', (img.width, img.height), img.rgb))
    def selectStart(self, event):
        self.is_selecting = True
        self.area.setStartPoint(self.mouse.position[0],self.mouse.position[1])

    def changeSelectionArea(self, event):
        if self.is_selecting:
            self.area.updateEndPoint(self.mouse.position[0],self.mouse.position[1],self.startW,self.startH)

    def selectDone(self, event):
        self.is_selecting = False
        self.captureImage()
        self.exit(None)
            


#FMatPix界面
class MainWindow:
    def __init__(self, image_path=None, ocr=None, detector=None):
        #设置缩放因子
        self.root = tk.Tk()
        self.root.tk.call('tk', 'scaling', sf)
        #默认窗口最前，如果最小化，则取消窗口最前；最大化，则恢复窗口最前。
        self.root.attributes("-topmost",1)
        self.root.bind('<Map>',lambda x:self.root.attributes("-topmost",1))
        self.root.bind('<Unmap>',lambda x:self.root.attributes("-topmost",0))
        # 固定窗口尺寸
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
        self.history_window = None
        self.historyImgs = []
        self.isBatch = False
        self.isPage = False
        self.pagePhotos = {}
        self.photoId = 0
        self.autocopylatex = False
        self.autocopymathml = False
        if image_path is None:
            raise ValueError("Image path must be provided.")

        self.root.title("Free Formula OCR for Latex and MathML v1.9 -~- Ltc")
        self.root.geometry(f"{int(630*pow(screen_scale_rate,0.7))}x{int(60*pow(screen_scale_rate,0.7))}")
        #控制同时只能有一个Screen
        self.hasScreen = False
        #初次运行读取预设图片
        if getattr(sys, "frozen", None):
            basedir = sys._MEIPASS
        else:
            basedir = os.path.dirname(__file__)
        self.image = Image.open(os.path.join(basedir,"models/GuiStartImg.PNG"))
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
            self.button_frame, text="History", width=12, height=2, command=self.create_table
        )
        self.history_button.grid(row=0, column=3, padx=5, pady=5)
        
        self.batch_button = tk.Button(
            self.button_frame, text="Batch Ocr", width=12, height=2, command=self.batchRecognition
        )
        self.batch_button.bind("<Double-Button-1>", self.on_doubleclick_batch)
        self.batch_button.grid(row=0, column=4, padx=5, pady=5)
        self.page_button = tk.Button(
            self.button_frame, text="Page Ocr", width=12, height=2)
        self.page_button.bind("<Double-Button-1>", self.on_doubleclick_page)
        self.page_button.grid(row=0, column=5, padx=5, pady=5)
        self.help_button = tk.Button(
            self.button_frame, text="Help", width=12, height=2, command=self.on_help
        )
        self.help_button.grid(row=0, column=6, padx=5, pady=5)
    

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
    
    #创建历史公式显示界面
    def create_table(self):
        if self.history_window and self.history_window.winfo_exists():
            self.history_window.destroy()
        history = self.history
        scrollSum = 0
        def on_left_click(event,index):
            pyperclip.copy(history[0][index])
        def on_right_click(event,index):
            pyperclip.copy(history[1][index])
        def on_mousewheel(event):
            scrollDis = int(-1 * (event.delta / 120))
            topPos = canvas.yview()[0]
            if topPos==0.0 and scrollDis==-1:
                return
            canvas.yview_scroll(scrollDis, "units")
        def on_window_close():
            self.history_window.destroy()
            self.history_window = None  # 窗口关闭时重置为None
            
        table = tk.Toplevel(self.root)
        self.history_window = table  # 更新跟踪属性
        table.protocol("WM_DELETE_WINDOW", on_window_close)  # 绑定关闭事件
        table.title('左键单击图片复制tex，右键单击图片复制mathml')
        table.attributes("-topmost",1)
        table.bind('<Map>',lambda x:table.attributes("-topmost",1))
        table.bind('<Unmap>',lambda x:table.attributes("-topmost",0))
        table.geometry("470x500")
        rootX = self.root.winfo_x()
        rootY = self.root.winfo_y()
        rootH = self.root.winfo_height()
        tableX = rootX
        tableY = rootY + rootH + 30
        table.geometry(f"+{tableX}+{tableY}")
        # 创建 Canvas 和 Scrollbar，实现滚动查看功能
        canvas = tk.Canvas(table, borderwidth=0, background="#ffffff")
        scrollbar = tk.Scrollbar(table, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, background="#ffffff")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        button_frame = tk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=5)
        copyall_button = tk.Button(button_frame,text='Copy All',width=12,height=1,command=self.copy_history_copy)
        clearall_button = tk.Button(button_frame,text='Clear All',width=12,height=1,command=self.on_history_refresh)
        copyall_button.grid(row=0, column=0, padx=65, pady=5)
        clearall_button.grid(row=0, column=1, padx=65, pady=5)
        
        for index in range(len(self.history[0])):
            # 创建一个 Frame 作为一行
            row_frame = tk.Frame(scrollable_frame)
            row_frame.pack(fill=tk.X, pady=5)
            # 显示字符串
            text_label = tk.Label(row_frame, text=str(index+1), font=("Arial", 12),justify="center", anchor="center")
            text_label.pack(side=tk.LEFT, padx=5)
            # 显示图像
            w0,h0 = self.historyImgs[index].size[0],self.historyImgs[index].size[1]
            w1 = 400
            h1 = int(h0*w1/w0)
            img = ImageTk.PhotoImage(self.historyImgs[index].resize((w1,h1)))
            image_label = tk.Label(row_frame, image=img)
            image_label.image = img  # 保持对图像的引用，防止被垃圾回收
            image_label.pack(side=tk.RIGHT, padx=5)
            image_label.bind('<Button-1>',lambda event,i=index:on_left_click(event,i))
            image_label.bind('<Button-3>',lambda event,i=index:on_right_click(event,i))

            
    #复制历史公式，同时有Latex和Mathml
    def copy_history_copy(self):
        #复制self.results到clipboard
        if self.history==[[],[]]:
            pyperclip.copy('no history')
            return
        out = ''
        hislen = len(self.history[0])
        for i,res in enumerate(self.history[0]):
            out += '['+str(i+1)+']'+res+'\n'
        out += '\n\n\n\n\n'
        for i,res in enumerate(self.history[1]):
            out += '['+str(i+1)+']'+res+'\n'
        pyperclip.copy(out)
       
    #清空历史公式
    def on_history_refresh(self):
        print('清空历史公式')
        self.history = [[],[]]
        self.historyImgs = []
        
    #双击batch button，开启或关闭批量识别模式：
    def on_doubleclick_batch(self,event):
        global capturedImgs
        if self.isBatch == True:
            self.isBatch = False
            self.batch_button.config(bg='SystemButtonFace')
            capturedImgs = []
            return
        if self.isPage == True:
            self.isPage = False
            self.page_button.config(bg='SystemButtonFace')
        self.isBatch = True
        self.batch_button.config(bg='DarkGray')
        capturedImgs = []

    #双击page button，开启或关闭整页识别+显示模式：
    def on_doubleclick_page(self,event):
        global capturedImgs
        if self.isPage == True:
            self.isPage = False
            self.page_button.config(bg='SystemButtonFace')
            capturedImgs = []
            return
        if self.isBatch == True:
            self.isBatch = False
            capturedImgs = []
            self.batch_button.config(bg='SystemButtonFace')
            self.progress_label.config(text='')

        self.isPage = True
        self.page_button.config(bg='DarkGray')
        capturedImgs = []

    #核心识别函数
    def cal(self,img):
        try:
            print('识别')
            imgs,boxes,alignment = self.detector.detect_image(img)
            print('检测公式数量')
            print(len(imgs))
            results = self.ocr.predict(imgs)
            latextext = utils.gatherOcrResults(results,alignment)
            print('转换')
            tmp1 = utils.InvokeTexmml(latextext)
            print('后处理')
            mathtypetext = utils.postprocessForMathml(tmp1)
            return (latextext,mathtypetext,boxes)
        except Exception as e:
            print("核心识别函数出错")
            print(e)
            return ('error','error')
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
            self.historyImgs.append(img)
            if self.autocopylatex:
                self.on_copy_latex()
            if self.autocopymathml:
                self.on_copy_mathtype()
        except queue.Empty as e:
            print("识别结果为空")
            pass

    def batchRecognition(self):
        global capturedImgs
        if len(capturedImgs)==0:
            return
        try:
            print('状态灯更新为红，显示剩余待识别公式数量')
            self.progress_label.config(background='red',text=str(len(capturedImgs)))
            self.root.update()
            for i in range(len(capturedImgs)):
                t1 = time.time()
                self.runCalWithTimeout(self.cal,capturedImgs[i],self.timeout)
                self.progress_label.config(text=str(len(capturedImgs)-i-1))
                self.root.update()
                print('检测+识别共用时：'+ str(time.time()-t1)+ 's')
        except Exception as e:
            print(e)
            print('batch recognition error')
        finally:
            capturedImgs = []
            print('状态灯更新为绿，取消显示剩余待识别公式数量')
            self.progress_label.config(background='green',text='')
            self.root.update()
    
    def pageCal(self,img,event):
        try:
            print('识别')
            imgs,boxes,alignment = self.detector.detect_image(img)
            print('检测公式数量')
            print(len(imgs))
            texResults = self.ocr.predict(imgs)
            mmlResults = [utils.postprocessForMathml(utils.InvokeTexmml(r)) for r in texResults]
            return (texResults,mmlResults,boxes,img)
        except Exception as e:
            print("核心识别函数出错")
            print(e)
            event.set()
            return ([],[],[],img)
        finally:
            event.set()
            
    def pageRecognition(self):
        # 绘制矩形框
        def draw_boxes(canvas, boxes,ratio):
            for i, box in enumerate(boxes):
                x, y, w, h = box
                x1 = int(x * ratio)
                y1 = int(y * ratio)
                x2 = int((x + w) * ratio)
                y2 = int((y + h) * ratio)
                canvas.create_rectangle(x1, y1, x2, y2, outline="red", fill="", stipple="gray50", width=2)


        #展示结果
        def afterCal(res):
            img = res[3]
            boxes = res[2]
            texres = res[0]
            mmlres = res[1]
            width, height = img.size
            new_width = 550
            ratio = new_width / width
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height))
            width, height = img.size

            
            table = tk.Toplevel(self.root)
            table.title('左键单击复制tex，右键单击复制mml')
            if new_height>700:
                table_height = 700
            else:
                table_height = new_height
            table.geometry(f"{new_width+20}x{table_height}")
            table.attributes("-topmost",1)
            table.bind('<Map>',lambda x:table.attributes("-topmost",1))
            table.bind('<Unmap>',lambda x:table.attributes("-topmost",0))
            rootX = self.root.winfo_x()
            rootY = self.root.winfo_y()
            rootH = self.root.winfo_height()
            tableX = rootX
            tableY = rootY + rootH + 30
            table.geometry(f"+{tableX}+{tableY}")

            
            canvas = tk.Canvas(table, width=new_width, height=new_height)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar = tk.Scrollbar(table, orient=tk.VERTICAL, command=canvas.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            canvas.configure(yscrollcommand=scrollbar.set)
            # 左键单击复制tex
            def on_left_click(event,boxes,texres,ratio):
                x_offset = canvas.xview()[0] * new_width
                y_offset = canvas.yview()[0] * new_height
                #print(canvas.yview()[0],y_offset,event.y+2*y_offset)
                x, y = event.x + x_offset, event.y + y_offset
                for i, box in enumerate(boxes):
                    bx, by, bw, bh = box
                    print(bx,x/ratio,bx+bw,by,y/ratio,by+bh)
                    if bx <= x/ratio < bx + bw and by <= y/ratio < by + bh:
                        print('复制tex')
                        pyperclip.copy(texres[i])
                print('\n')
            # 右键单击复制mml
            def on_right_click(event,boxes,mmlres,ratio):
                x_offset = canvas.xview()[0] * new_width
                y_offset = canvas.yview()[0] * new_height
                x, y = event.x+x_offset, event.y+y_offset
                for i, box in enumerate(boxes):
                    bx, by, bw, bh = box
                    print(bx,x/ratio,bx+bw,by,y/ratio,by+bh)
                    if bx <= x/ratio < bx + bw and by <= y/ratio < by + bh:
                        #点击到公式，复制至剪切板
                        print('复制mml')
                        pyperclip.copy(mmlres[i])
            def release_image(photoid):
                # 从 self.images 列表中移除对应的图像引用
                if photoid in self.pagePhotos:
                    try:
                        self.pagePhotos.pop(photoid)
                        print("Image reference released.")
                    except IndexError:
                        print("Image index out of range.")
                    finally:
                        table.destroy()
                else:
                    print("No valid image index found.")
            canvas.config(scrollregion=(0, 0, new_width, new_height))
            nowid = self.photoId
            table.protocol("WM_DELETE_WINDOW", lambda:release_image(nowid))
            self.pagePhotos[nowid] = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor=tk.NW, image=self.pagePhotos[nowid])
            self.photoId += 1
            draw_boxes(canvas, boxes, ratio)
            canvas.bind("<Button-1>", lambda event: on_left_click(event,boxes,texres,ratio))
            canvas.bind("<Button-3>", lambda event: on_right_click(event,boxes,mmlres,ratio))
        def calAndShow(img,event):
            res = self.pageCal(img,event)
            self.root.after(0,lambda: afterCal(res))

        global capturedImgs
        print('pageOcr')
        if len(capturedImgs)==0:
            return
        try:
            print('状态灯更新为红，整页公式识别中')
            self.progress_label.config(background='red')
            self.root.update()
            time.sleep(0.5)
            event = threading.Event()
            thread = threading.Thread(target=calAndShow,args=(capturedImgs[0],event))
            thread.start()
            event.wait()
        except Exception as e:
            print(e)
            print('page recognition error')
        finally:
            capturedImgs = []
            print('状态灯更新为绿，整页识别完成')
            self.progress_label.config(background='green')
            self.root.update()

    
    #打开截图窗口并识别
    def on_snip(self,snip=None):
        #禁止同时出现两个截图窗口，或在状态灯为红色时截图
        if self.hasScreen==True or self.progress_label.cget('bg')=='red':
            return
        self.hasScreen = True
        global capturedImgs
        print("执行开始")
        try:
            print('截图')
            ScreenShot()
            if self.isBatch:
                self.progress_label.config(text=str(len(capturedImgs)))
                return
            if self.isPage:
                self.pageRecognition()

            print('状态灯更新为红')
            self.progress_label.config(background='red')
            self.root.update()
            print('开始计算')
            if len(capturedImgs)==1:
                t1 = time.time()
                self.runCalWithTimeout(self.cal,capturedImgs[0],self.timeout)
                print('检测+识别共用时：'+ str(time.time()-t1)+ 's')
            else:
                print('image==None')
            self.hasScreen = False
            print('执行完成')
        except Exception as e:
            print(e)
            print('recognition error')
        finally:
            if not self.isBatch and not self.isPage and len(capturedImgs)==1:
                capturedImgs.pop()
            self.hasScreen = False
            self.progress_label.config(background='green')
            self.root.update()
    
    def on_help(self):
        top = tk.Toplevel()
        top.title("Help")
        top.attributes("-topmost",1)
        msg = tk.Message(
            top,
            width=600,
            font=("TkDefaultFont", 14),
            text="""1、按Alt+Q开始截图，鼠标左键选择区域，Esc键退出。
2、完成识别后，可单击copy latex/mathml来复制本次识别的公式。
3、双击copy latex/mathml，来开启或关闭识别后自动复制功能。
4、单击History显示历史公式图片，左单击图片复制Tex，右单击图片复制MathMl；
   copy all复制全部公式到剪切板，建议粘贴到文本编辑器(例如记事本)查看，不要
   直接复制到mathtype查看(数量太多会报错)；clear all清空全部历史公式。
5、红色状态灯表示正在识别，绿色状态灯表示识别完成。
6、双击Batch Ocr开启或关闭批量识别模式；
   批量识别模式：每次截图后不识别，累积多个截图，数量显示在绿色状态灯；
   单击Batch Ocr开始批量识别，剩余待识别公式数量显示在红色状态灯；
   批量识别完成后，点击History，查看、复制公式。
7、双击Page Ocr开始或关闭整页识别模式
   整页识别模式：截一张大图，检测、识别、显示其中全部公式，单击复制
   注意：整页识别模型的结果不计入历史，且不能和批量识别模式同时开启
8、满意的话请到Github给个赞，后续改进也会在上面发布：
   https://github.com/ai25395/FMatPix """,
        )
        msg.pack()

    def run(self):
        self.root.mainloop()

#截图键检测
def on_press(key):
    if window.hasScreen:
        return
    try:
        if key == keyboard.Key.alt or key == keyboard.Key.alt_l or key == keyboard.Key.alt_r or key == keyboard.Key.alt_gr:
            s.add('alt')
        if key.char == 'q':
            s.add('q')
    except AttributeError:
        pass 
def on_release(key):
    try:
        if key == keyboard.Key.alt or key == keyboard.Key.alt_l or key == keyboard.Key.alt_r or key == keyboard.Key.alt_gr:
            if 'alt' in s:
                s.remove('alt')
        if key.char == 'q':
            if 'q' in s:
                s.remove('q')
    except AttributeError as e:
        pass 
    
def start_keyboard_listener():
    with keyboard.Listener(on_press=on_press,on_release=on_release) as listener:
        listener.join()    
def checkqueue():
    try:
        if (s.__contains__('alt') and s.__contains__('q')):
            window.on_snip()
            s.clear()
    except queue.Empty:
        pass
    window.root.after(50, checkqueue)
    
if __name__ == "__main__":
    if getattr(sys, "frozen", None):
        basedir = sys._MEIPASS
    else:
        basedir = os.path.dirname(__file__)
    detector = utils.PreProcess(os.path.join(basedir, "models/yolo-detect.onnx"))
    model = models.OcrModel()
    window = MainWindow(os.path.join(basedir, "models/GuiStartImg.png"), model, detector)
    #消息队列，用于键盘监听线程和主程序通信
    s = set() 
    #启动键盘监听线程
    listener_process = threading.Thread(target=start_keyboard_listener,daemon=True)
    listener_process.start()
    window.root.after(50, checkqueue)
    #启动主程序
    window.root.mainloop()
        

