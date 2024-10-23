import cv2
import numpy as np
from PIL import Image
import traceback
from pathlib import Path
from typing import List
from typing import Tuple, Union
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
import os
import sys
import codecs
import re

def remove_spaces_in_tag(text):  
    # 正则表达式模式：匹配固定的开头</、任意数量的空格、任意字符（非贪婪，作为标签名）、固定的结尾>  
    # 捕获组1：标签名（可能包含空格）  
    pattern = r'</\s*(.*?)\s*>'  
      
    # 回调函数，用于处理每个匹配项，去除标签名内的空格  
    def fix_match(match):  
        # 提取标签名（带空格），并去除空格  
        tag_name = match.group(1).replace(' ','')
        # 返回修复后的字符串，注意这里我们重新构造了</和>  
        return f'</{tag_name}>'  
      
    # 使用re.sub和回调函数替换所有匹配项  
    fixed_text = re.sub(pattern, fix_match, text)  
      
    return fixed_text 

def remove_spaces_in_quotes(text):  
    # 使用正则表达式查找所有双引号括起来的内容  
    pattern = r'"([^"]*)"'  
      
    # 使用 re.sub() 函数进行替换，回调函数用于处理每个匹配项  
    def replace_match(match):  
        # match.group(1) 获取双引号之间的内容（即第一个捕获组）  
        quoted_text = match.group(1)  
        # 去除引号内内容的空格并返回新的字符串  
        return '"' + quoted_text.replace(' ', '') + '"'  
      
    # 执行替换操作  
    result = re.sub(pattern, replace_match, text)  
    return result

def preprocessForMathml(mathml):
    #处理右下角标和正下角标混淆的问题
    # 定义正则表达式模式，用来匹配\operatorname*{...}格式的内容
    pattern = r'\\operatorname\*\{([^}]*)\}'
    # 使用re.sub()函数进行替换
    # \1 是一个反向引用，它代表了第一个括号内的匹配内容
    mathml = re.sub(pattern, r'\\mathop{\1}\\limits', mathml)
    return mathml

def postprocessForMathml(mathml):
    # 处理大括号多行公式问题
    if mathml.__contains__(
        r'<mo stretchy="true" fence="true" form="prefix">&#x0007B;</mo>'
    ):
        mathml = mathml.replace(
            r'<mo stretchy="true" fence="true" form="prefix">&#x0007B;</mo>',
            r'<mfenced close="" open="{">',
        )
        mathml = mathml.replace(
            r'<mo stretchy="true" fence="true" form="postfix" />', r"</mfenced>"
        )
        

    # 处理英文字母异形体问题，此方法不完美，后续待完善
    if mathml.__contains__('&#x'):
        # 定义正则表达式模式  
        pattern = r'<mi>&#x([0-9A-Fa-f]+);</mi>'  
  
        # 定义一个函数来检查Unicode码点是否在指定范围内  
        def is_in_range_bold(code_point_hex):  
            try:  
                # 将十六进制字符串转换为整数  
                code_point = int(code_point_hex, 16)  
                # 检查是否在指定范围内  
                return 0x1D400 <= code_point <= 0x1D433  
            except ValueError:  
                # 如果转换失败（理论上不应该发生，因为我们已经通过正则表达式匹配了十六进制数字）  
                return False  
        # 定义一个函数来检查Unicode码点是否在指定范围内  
        def is_in_range_doublestruck(code_point_hex):  
            try:  
                # 将十六进制字符串转换为整数  
                code_point = int(code_point_hex, 16)  
                # 检查是否在指定范围内  
                return 0x1D538 <= code_point <= 0x1D56B
            except ValueError:  
                # 如果转换失败（理论上不应该发生，因为我们已经通过正则表达式匹配了十六进制数字）  
                return False  
        def is_in_range_script(code_point_hex):
            try:  
                # 将十六进制字符串转换为整数  
                code_point = int(code_point_hex, 16)  
                # 检查是否在指定范围内  
                return 0x1D49C <= code_point <= 0x1D537
            except ValueError:  
                # 如果转换失败（理论上不应该发生，因为我们已经通过正则表达式匹配了十六进制数字）  
                return False
        symboldict = {}
        specialUnicodes = [[0x1D400,0x1D433],[0x1D538,0x1D56B],[0x1D49C,0x1D537]]
        count = 0
        chcs = {  
                0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',  
                10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',  
                20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'a', 27: 'b', 28: 'c', 29: 'd',  
                30: 'e', 31: 'f', 32: 'g', 33: 'h', 34: 'i', 35: 'j', 36: 'k', 37: 'l', 38: 'm', 39: 'n',  
                40: 'o', 41: 'p', 42: 'q', 43: 'r', 44: 's', 45: 't', 46: 'u', 47: 'v', 48: 'w', 49: 'x',  
                50: 'y', 51: 'z'  }
        for su in specialUnicodes:
            st = su[0]
            end = su[1]
            for x in range(st,end+1):
                symboldict[(hex(x)[2:]).zfill(5).upper()] = chcs[count%52]
                count += 1
        # 定义替换函数  
        def conditional_replacement(match):  
            hex_value = match.group(1)  # 获取第一个捕获组的内容  
            if is_in_range_bold(hex_value):  
                # 如果在范围内，进行替换  
                return "<mstyle mathvariant='bold' mathsize='normal'><mi>" + symboldict[hex_value] + "</mi></mstyle>"
            elif is_in_range_doublestruck(hex_value):
                return "<mi mathvariant='double-struck'>" + symboldict[hex_value]+"</mi>"
            elif is_in_range_script(hex_value):
                return "<mi mathvariant='script'>" + symboldict[hex_value]+"</mi>"
            else:  
                # 如果不在范围内，不替换（返回原始匹配项）  
                return match.group(0)  
        mathml = re.sub(pattern, conditional_replacement, mathml)

    return mathml

def token2str(tokens, tokenizer) -> list:
    if len(tokens.shape) == 1:
        tokens = tokens[None, :]
    dec = [tokenizer.decode(tok, clean_up_tokenization_spaces=False) for tok in tokens]
    return [
        "".join(detok.split(" "))
        .replace("Ġ", " ")
        .replace("Ċ", " ")
        .replace("[EOS]", "")
        .replace("[BOS]", "")
        .replace("[PAD]", "")
        .strip()
        for detok in dec
    ]


class PreProcess:
    def __init__(self, detect_path, max_dims: List[int], min_dims: List[int]):
        self.max_dims, self.min_dims = max_dims, min_dims
        self.mean = np.array([0.7931, 0.7931, 0.7931]).astype(np.float32)
        self.std = np.array([0.1738, 0.1738, 0.1738]).astype(np.float32)

    def letterbox(
        self,
        im,
        new_shape=(128, 640),
        color=(114, 114, 114),
        auto=False,
        scaleFill=False,
        scaleup=True,
        stride=32,
    ):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        # if shape[0] > shape[1]:
        #     max_wh = max(new_shape[0], new_shape[1])
        #     new_shape = (max_wh, max_wh)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
        # return im, ratio, (dw, dh)

        return im

    def pad(self, img: Image.Image, divable: int = 32) -> Image.Image:
        """Pad an Image to the next full divisible value of `divable`. Also normalizes the image and invert if needed.

        Args:
            img (PIL.Image): input image
            divable (int, optional): . Defaults to 32.

        Returns:
            PIL.Image
        """
        threshold = 128
        data = np.array(img.convert("LA"))
        if data[..., -1].var() == 0:
            data = (data[..., 0]).astype(np.uint8)
        else:
            data = (255 - data[..., -1]).astype(np.uint8)

        data = (data - data.min()) / (data.max() - data.min()) * 255
        if data.mean() > threshold:
            # To invert the text to white
            gray = 255 * (data < threshold).astype(np.uint8)
        else:
            gray = 255 * (data > threshold).astype(np.uint8)
            data = 255 - data

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        rect = data[b : b + h, a : a + w]
        im = Image.fromarray(rect).convert("L")
        dims: List[Union[int, int]] = []
        for x in [w, h]:
            div, mod = divmod(x, divable)
            dims.append(divable * (div + (1 if mod > 0 else 0)))

        padded = Image.new("L", tuple(dims), 255)
        padded.paste(im, (0, 0, im.size[0], im.size[1]))
        return padded

    def minmax_size(
        self,
        img: Image.Image,
    ) -> Image.Image:
        """Resize or pad an image to fit into given dimensions

        Args:
            img (Image): Image to scale up/down.

        Returns:
            Image: Image with correct dimensionality
        """
        if self.max_dims is not None:
            ratios = [a / b for a, b in zip(img.size, self.max_dims)]
            if any([r > 1 for r in ratios]):
                size = np.array(img.size) // max(ratios)
                img = img.resize(size.astype(int), Image.BILINEAR)

        if self.min_dims is not None:
            padded_size: List[Union[int, int]] = [
                max(img_dim, min_dim)
                for img_dim, min_dim in zip(img.size, self.min_dims)
            ]

            new_pad_size = tuple(padded_size)
            if new_pad_size != img.size:  # assert hypothesis
                padded_im = Image.new("L", new_pad_size, 255)
                padded_im.paste(img, img.getbbox())
                img = padded_im
        return img

    def normalize(self, img: np.ndarray, max_pixel_value=255.0) -> np.ndarray:
        mean = self.mean * max_pixel_value
        std = self.std * max_pixel_value
        denominator = np.reciprocal(std, dtype=np.float32)
        img = img.astype(np.float32)
        img -= mean
        img *= denominator
        return img

    @staticmethod
    def to_gray(img) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def transpose_and_four_dim(img: np.ndarray) -> np.ndarray:
        return img.transpose(2, 0, 1)[:1][None, ...]


class OrtInferSession:
    def __init__(self, model_path: Union[str, Path], num_threads: int = 6):
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

    def __call__(self, input_content: List[np.ndarray]) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            return self.session.run(None, input_dict)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e

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
    def verify_exist(model_path: Union[Path, str]):
        if not isinstance(model_path, Path):
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exist!")

        if not model_path.is_file():
            raise FileExistsError(f"{model_path} must be a file")


class Decoder:
    def __init__(self, decoder_path: Union[Path, str], max_seq_len=512):
        self.max_seq_len = max_seq_len
        self.session = OrtInferSession(decoder_path)

    def __call__(
        self,
        start_tokens,
        seq_len=256,
        eos_token=None,
        temperature=0.00001,
        filter_thres=0.9,
        context=None,
    ):
        num_dims = len(start_tokens.shape)

        b, t = start_tokens.shape

        out = start_tokens

        confidences = []
        for _ in range(seq_len):
            x = out[:, -self.max_seq_len :]
            ort_outs = self.session([x.astype(np.int64), context])[0]
            np_preds = ort_outs
            np_logits = np_preds[:, -1, :]
            sample = np.argmax(np_logits, axis=-1)
            np_probs = self.softmax(np_logits, axis=-1)
            confidence = np.max(np_probs, axis=1)
            confidences.append(confidence)
            sample = np.expand_dims(sample, axis=-1)
            out = np.concatenate([out, sample], axis=-1)

            if (
                eos_token is not None
                and (np.cumsum(out == eos_token, axis=1)[:, -1] >= 1).all()
            ):
                break

        out = out[:, t:]
        if num_dims == 1:
            out = out.squeeze(0)
        return (out, [round(np.mean(i) * 100, 2) for i in confidences])

    @staticmethod
    def softmax(x, axis=None) -> float:
        def logsumexp(a, axis=None, b=None, keepdims=False):
            a_max = np.amax(a, axis=axis, keepdims=True)

            if a_max.ndim > 0:
                a_max[~np.isfinite(a_max)] = 0
            elif not np.isfinite(a_max):
                a_max = 0

            tmp = np.exp(a - a_max)

            # suppress warnings about log of zero
            with np.errstate(divide="ignore"):
                s = np.sum(tmp, axis=axis, keepdims=keepdims)
                out = np.log(s)

            if not keepdims:
                a_max = np.squeeze(a_max, axis=axis)
            out += a_max
            return out

        return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

    def npp_top_k(self, logits, thres=0.9):
        k = int((1 - thres) * logits.shape[-1])
        val, ind = self.np_top_k(logits, k)
        probs = np.full_like(logits, float("-inf"))
        np.put_along_axis(probs, ind, val, axis=1)
        return probs

    @staticmethod
    def np_top_k(
        a: np.ndarray, k: int, axis=-1, largest=True, sorted=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        if axis is None:
            axis_size = a.size
        else:
            axis_size = a.shape[axis]

        assert 1 <= k <= axis_size

        a = np.asanyarray(a)
        if largest:
            index_array = np.argpartition(a, axis_size - k, axis=axis)
            topk_indices = np.take(index_array, -np.arange(k) - 1, axis=axis)
        else:
            index_array = np.argpartition(a, k - 1, axis=axis)
            topk_indices = np.take(index_array, np.arange(k), axis=axis)

        topk_values = np.take_along_axis(a, topk_indices, axis=axis)
        if sorted:
            sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
            if largest:
                sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
            sorted_topk_values = np.take_along_axis(
                topk_values, sorted_indices_in_topk, axis=axis
            )
            sorted_topk_indices = np.take_along_axis(
                topk_indices, sorted_indices_in_topk, axis=axis
            )
            return sorted_topk_values, sorted_topk_indices
        return topk_values, topk_indices

    @staticmethod
    def multinomial(weights, num_samples, replacement=True):
        weights = np.asarray(weights)
        weights /= np.sum(weights)  # 确保权重之和为1
        indices = np.arange(len(weights))
        samples = np.random.choice(
            indices, size=num_samples, replace=replacement, p=weights
        )
        return samples


class ONNXRuntimeError(Exception):
    pass
