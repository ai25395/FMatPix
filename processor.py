from typing import Dict, Union, Optional, List
from transformers import DonutImageProcessor, DonutProcessor, AutoImageProcessor
from transformers.image_processing_utils import get_size_dict, BatchFeature
from transformers.image_transforms import pad
from transformers.image_utils import PILImageResampling, ImageInput, ChannelDimension, make_list_of_images, valid_images, to_numpy_array,get_image_size
import numpy as np
import PIL
from transformers import DonutSwinConfig, VisionEncoderDecoderConfig

IMAGE_STD = [0.229, 0.224, 0.225]
IMAGE_MEAN = [0.485, 0.456, 0.406]
maxheight = 420
maxwidth = 420
maxtokens = 384
def get_config(model_checkpoint):
    config = VisionEncoderDecoderConfig.from_pretrained(model_checkpoint)
    encoder_config = vars(config.encoder)
    encoder = VariableDonutSwinConfig(**encoder_config)
    config.encoder = encoder
    return config
class VariableDonutSwinConfig(DonutSwinConfig):
    pass

def load_processor(processorPath):
    AutoImageProcessor.register(VariableDonutSwinConfig, slow_image_processor_class=VariableDonutImageProcessor)
    processor = VariableDonutProcessor.from_pretrained(processorPath)
    processor.image_processor.max_size = {"height": maxheight, "width": maxwidth}
    processor.image_processor.size = [maxheight, maxwidth]
    processor.image_processor.image_mean = IMAGE_MEAN
    processor.image_processor.image_std = IMAGE_STD
    processor.image_processor.train = False

    processor.tokenizer.model_max_length = maxtokens
    processor.train = False
    return processor


class VariableDonutImageProcessor(DonutImageProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def numpy_resize(self, image: np.ndarray, size, resample):
        image = PIL.Image.fromarray(image)
        resized = self.pil_resize(image, size, resample)
        resized = np.array(resized, dtype=np.uint8)
        resized_image = resized.transpose(2, 0, 1)

        return resized_image

    def pil_resize(self, image: PIL.Image.Image, size, resample):
        width, height = image.size
        max_width, max_height = size["width"], size["height"]
        if width != max_width or height != max_height:
            # Shrink to fit within dimensions
            width_scale = max_width / width
            height_scale = max_height / height
            scale = min(width_scale, height_scale)

            new_width = min(int(width * scale), max_width)
            new_height = min(int(height * scale), max_height)

            image = image.resize((new_width, new_height), resample)

        image.thumbnail((max_width, max_height), resample)

        assert image.width <= max_width and image.height <= max_height

        return image

    def process_inner(self, images: List[List], train=False):
        # This will be in list of lists format, with height x width x channel
        assert isinstance(images[0], (list, np.ndarray))

        # convert list of lists format to array
        if isinstance(images[0], list):
            # numpy unit8 needed for augmentation
            np_images = [np.array(img, dtype=np.uint8) for img in images]
        else:
            np_images = [img.astype(np.uint8) for img in images]

        assert np_images[0].shape[2] == 3 # RGB input images, channel dim last

        # This also applies the right channel dim format, to channel x height x width
        np_images = [self.numpy_resize(img, self.max_size, self.resample) for img in np_images]
        assert np_images[0].shape[0] == 3 # RGB input images, channel dim first

        # Convert to float32 for rescale/normalize
        np_images = [img.astype(np.float32) for img in np_images]

        # Pads with 255 (whitespace)
        # Pad to max size to improve performance
        max_size = self.max_size
        np_images = [
            self.pad_image(
                image=image,
                size=max_size,
                random_padding=train, # Change amount of padding randomly during training
                input_data_format=ChannelDimension.FIRST,
                pad_value=255.0
            )
            for image in np_images
        ]

        # Rescale and normalize
        np_images = [
            self.rescale(img, scale=self.rescale_factor, input_data_format=ChannelDimension.FIRST)
            for img in np_images
        ]
        np_images = [
            self.normalize(img, mean=self.image_mean, std=self.image_std, input_data_format=ChannelDimension.FIRST)
            for img in np_images
        ]

        return np_images

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
        do_pad: bool = None,
        random_padding: bool = False,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> PIL.Image.Image:
        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # Convert to numpy for later processing steps
        images = [to_numpy_array(image) for image in images]

        images = self.process_inner(images, train=False)

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    def pad_image(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        random_padding: bool = False,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        pad_value: float = 0.0,
    ) -> np.ndarray:
        output_height, output_width = size["height"], size["width"]
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)

        delta_width = output_width - input_width
        delta_height = output_height - input_height

        assert delta_width >= 0 and delta_height >= 0

        if random_padding:
            pad_top = np.random.randint(low=0, high=delta_height + 1)
            pad_left = np.random.randint(low=0, high=delta_width + 1)
        else:
            pad_top = delta_height // 2
            pad_left = delta_width // 2

        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        return pad(image, padding, data_format=data_format, input_data_format=input_data_format, constant_values=pad_value)


class VariableDonutProcessor(DonutProcessor):
    def __init__(self, image_processor=None, tokenizer=None, train=False, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self._in_target_context_manager = False
        self.train = train

    def __call__(self, *args, **kwargs):
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        images = kwargs.pop("images", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            images = args[0]
            args = args[1:]

        if images is None:
            raise ValueError("You need to specify images to process.")

        inputs = self.image_processor(images, *args, **kwargs)
        return inputs
