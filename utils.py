import re
from ftfy import fix_text
import subprocess
import os
import sys
import subprocess
import cv2
from PIL import Image
import numpy as np
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
#使用Texmml这个Js库完成Tex -> MML
def InvokeTexmml(latextext):
    if getattr(sys, "frozen", None):
        basedir = sys._MEIPASS
    else:
        basedir = os.path.dirname(__file__)
            
    nodePath = os.path.join(basedir, 'texmml/node.exe')
    jsscriptPath = os.path.join(basedir, 'texmml/tex2mml.js')
    resultPath = os.path.join(basedir,'texmml/result.txt')
    command = [nodePath, jsscriptPath, latextext, resultPath]
    
    #隐藏窗口
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE  # 隐藏窗口    
    # 启动子进程并等待其完成
    result = subprocess.run(command, check=True, startupinfo=startupinfo)
    # 检查子进程是否成功完成
    if result.returncode == 0:
        # 读取结果文件
        with open(resultPath, 'r', encoding='utf-8') as file:
            mml = file.read().strip()
            return mml
    else:
        return 'latex2mathmlError'
def postprocessForMathml(mathml):
    #上标横线问题
    if mathml.__contains__('‾'):
        mathml = mathml.replace('‾','-')
    #大括号问题
    if mathml.__contains__('<mo fence="true" form="prefix">{</mo>') and not mathml.__contains__('<mo fence="true" form="postfix">}</mo>'):
        mathml = mathml.replace('<mo fence="true" form="prefix">{</mo>','<mfenced close="" open="{">')
        mathml = mathml.replace('<mo fence="true" form="postfix"></mo>','</mfenced>')
    return mathml

def batch_inference(images, model, processor, temperature=0.0, max_tokens=384):
    try:
        images = [image.convert("RGB") for image in images]
        encodings = processor(images=images, return_tensors="pt", add_special_tokens=False)
        pixel_values = encodings["pixel_values"].to(model.dtype)
        pixel_values = pixel_values.to(model.device)
        additional_kwargs = {}
        if temperature > 0:
            additional_kwargs["temperature"] = temperature
            additional_kwargs["do_sample"] = True
            additional_kwargs["top_p"] = 0.95
        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_new_tokens=max_tokens,
            decoder_start_token_id=processor.tokenizer.bos_token_id,
            **additional_kwargs,
        )
        generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [postprocess(text) for text in generated_text]
    except Exception as e:
        print('推理错误')
        print(e)
    return generated_text

def gatherOcrResults(result,alignment):
    print('集成检测结果')
    if len(result)==1:
        return result[0]
    else:
        gathered = r'\begin{array}{'+ alignment+ '}'
        for r in result:
            gathered += r + r'\\'
        gathered = gathered[:-2] + r'\end{array}'
        return gathered
    
def remove_labels(text):
    pattern = r'\\label\{[^}]*\}'
    text = re.sub(pattern, '', text)
    ref_pattern = r'\\ref\{[^}]*\}'
    text = re.sub(ref_pattern, '', text)
    pageref_pattern = r'\\pageref\{[^}]*\}'
    text = re.sub(pageref_pattern, '', text)
    return text

def postprocess(text):
    text = fix_text(text)
    text = remove_labels(text)
    text = text.replace('$$','')
    return text

def determine_alignment(boxes, threshold=5):
    """
    判断矩形框是居中对齐、左对齐还是右对齐。

    :param boxes: 矩形框列表，每个元素为[x, y, w, h]
    :param threshold: 对齐判定的阈值
    :return: 返回矩形框的对齐方式：'左对齐'、'右对齐'或'居中对齐'
    """
    # 提取所有框的左边界（x坐标）和右边界（x + w坐标）
    left_edges = [x for x, _, w, _ in boxes]
    right_edges = [x + w for x, _, w, _ in boxes]
    try:
        # 计算左边界和右边界的最大和最小值，判断是否在阈值范围内
        left_range = max(left_edges) - min(left_edges)
        right_range = max(right_edges) - min(right_edges)
        centers = []
        for i in range(len(left_edges)):
            centers.append((left_edges[i]+right_edges[i])/2)
        center_range = max(centers) - min(centers)
        if abs(left_range-right_range)<center_range:
            return 'c'
        else:
            if left_range<right_range:
                return 'l'
            else:
                return 'r'
    except:
        return "c"
    
class OrtInferSession:
    def __init__(self, model_path, num_threads: int = 6):
        self.verify_exist(model_path)

        self.num_threads = num_threads
        self._init_sess_opt()

        EP_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        try:
            self.session = InferenceSession(
                str(model_path), sess_options=self.sess_opt, providers=EP_list
            )
        except TypeError:
            self.session = InferenceSession(str(model_path), sess_options=self.sess_opt)

    def _init_sess_opt(self):
        self.sess_opt = SessionOptions()
        self.sess_opt.log_severity_level = 4
        self.sess_opt.enable_cpu_mem_arena = False

        if self.num_threads != -1:
            self.sess_opt.intra_op_num_threads = self.num_threads

        self.sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    def __call__(self, input_content) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            return self.session.run(None, input_dict)
        except Exception as e:
            print('检测错误')
            print(e)

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_name(self, output_idx=0):
        return self.session.get_outputs()[output_idx].name

    def get_metadata(self):
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict

    @staticmethod
    def verify_exist(model_path):
        from pathlib import Path
        if not isinstance(model_path, Path):
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exist!")

        if not model_path.is_file():
            raise FileExistsError(f"{model_path} must be a file")
        
class PreProcess:
    def __init__(self, detect_path):
        self.max_dims, self.min_dims = [1024,512], [32,32]
        self.mean = np.array([0.7931, 0.7931, 0.7931]).astype(np.float32)
        self.std = np.array([0.1738, 0.1738, 0.1738]).astype(np.float32)
        self.detecter = OrtInferSession(detect_path)
    def detect_image(self, input_image):
        try:
            source_image = input_image
            original_image: np.ndarray = cv2.cvtColor(np.array(source_image),
                                                        cv2.COLOR_RGB2BGR)
            [height, width, _] = original_image.shape
            length = max((height, width))
            image = np.zeros((length, length, 3), np.uint8)
            image[0:height, 0:width] = original_image
            scale = length / 640
            image = cv2.resize(image, (640, 640))  # 调整图像大小
            image = image.astype(np.float32) / 255.0  # 归一化
            image = image.transpose(2, 0, 1)  # 重排数组顺序 HWC到CHW
            image = np.expand_dims(image, axis=0)  # 添加维度，以符合模型的输入
            outputs = self.detecter([image])[0].astype(np.float32)
            outputs = np.array([cv2.transpose(outputs[0])])  # 1 8400 6
            rows = outputs.shape[1]
            boxes = []
            scores = []
            class_ids = []
            for i in range(rows):
                classes_scores = outputs[0][i][4:]
                (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
                if maxScore >= 0.45:
                    box = [
                        outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                        outputs[0][i][2], outputs[0][i][3]]
                    boxes.append(box)
                    scores.append(maxScore)
                    class_ids.append(maxClassIndex)

            result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.45, 0.45, 0.5)
            sorted_result_boxes = sorted(result_boxes, key=lambda i: boxes[i][1] * scale)  # 根据NMS结果中的框的y坐标进行排序
            images = []
            for i in range(len(sorted_result_boxes)):
                index = sorted_result_boxes[i]
                box = boxes[index]
                x, y, w, h = round(box[0] * scale), round(box[1] * scale), round(box[2] * scale), round(box[3] * scale)
                cropped_image = source_image.crop((x, y, x + w, y + h))
                images.append(cropped_image)
            alignment = determine_alignment(boxes, threshold=5)
        except Exception as e:
            print('检测公式区域错误')
            print(e)
        return images, alignment

    