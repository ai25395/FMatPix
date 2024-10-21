import re
import time
import cv2
import numpy as np
import yaml
from PIL import Image
from transformers import PreTrainedTokenizerFast
import utils
from easydict import EasyDict
import os
import sys


class Latex_OCR:
    def __init__(
        self,
        encoder_path="models/encoder.onnx",
        config_path="models/config.yaml",
        decoder_path="models/decoder.onnx",
        tokenizer_path="models/tokenizer.json",
    ):
        if getattr(sys, "frozen", None):
            basedir = sys._MEIPASS
        else:
            basedir = os.path.dirname(__file__)
        decoder_path = os.path.join(basedir, decoder_path)
        tokenizer_path = os.path.join(basedir, tokenizer_path)
        config_path = os.path.join(basedir, config_path)
        with open(config_path, "r") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)
        self.args = EasyDict(args)
        self.max_dims = [
            self.args.get("max_width", 1024),
            self.args.get("max_height", 512),
        ]
        self.min_dims = [
            self.args.get("min_width", 32),
            self.args.get("min_height", 32),
        ]
        self.temperature = self.args.get("temperature", 0.00001)
        self.bos_token = self.args.get("bos_token")
        self.eos_token = self.args.get("eos_token")
        self.max_seq_len = self.args.get("max_seq_len", 512)
        self.pre_pro = utils.PreProcess(None, self.max_dims, self.min_dims)
        self.encoder = utils.OrtInferSession(encoder_path)
        self.decoder = utils.Decoder(decoder_path, self.max_seq_len)
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))

    def predict(self, image):
        t1 = time.time()
        images = [image]
        alignment = "c"
        final_imgs = []
        for idx, im in enumerate(images):
            image = cv2.cvtColor(
                np.array(self.pre_pro.pad(im, divable=1)), cv2.COLOR_BGR2RGB
            )
            image = self.pre_pro.letterbox(
                im=image, new_shape=[self.args["height"], self.args["width"]]
            )
            gray_img = self.pre_pro.to_gray(image)
            normal_img = self.pre_pro.normalize(gray_img).transpose(2, 0, 1)[:1]
            final_imgs.append(normal_img)
        x = np.stack(final_imgs, axis=0)
        ort_input_data = np.array([self.bos_token] * len(x))[:, None]
        context = self.encoder([x])[0].astype(np.float32)
        outputs, confidences = self.decoder(
            ort_input_data,
            self.max_seq_len,
            eos_token=self.eos_token,
            context=context,
            temperature=self.temperature,
        )
        formula = [
            self.post_process("".join(utils.token2str(output, self.tokenizer)).strip())
            for output in outputs
        ]
        res = {
            "formula": (
                formula[0]
                if len(formula) < 2
                else "\\begin{array}{"
                + alignment
                + "}"
                + " \\\\ ".join(formula)
                + "\end{array}"
            ),
            "confidence": str(round(np.mean(confidences), 2)) + "%",
        }
        res["elapse"] = "{:,}ms".format(int((time.time() - t1) * 1000))
        return res

    @staticmethod
    def post_process(s: str) -> str:
        """Remove unnecessary whitespace from LaTeX code.

        Args:
            s (str): Input string

        Returns:
            str: Processed image
        """
        text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
        letter = "[a-zA-Z]"
        noletter = r"[\W_^\d]"
        names = [x[0].replace(" ", "") for x in re.findall(text_reg, s)]
        s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
            news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
            if news == s:
                break
        # Replace \eqno with ~~~.
        s = re.sub(r"\\eqno", "~~~", s)
        return s
