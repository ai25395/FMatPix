import cv2
import numpy as np
from PIL import Image
import traceback
from pathlib import Path
from typing import List
from typing import Tuple, Union
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

def postprocessForMathml(mathml):
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
    # num_threads越多，识别越快
    def __init__(self, model_path: Union[str, Path], num_threads: int = 8):
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
