import argparse
import pickle
import sys
import cv2

from transforms import SamplerTransform
from transforms import ResizeTransform
from transforms import ReorderChannelsTransform
from transforms import ExpandTransform
from transforms import ImageLoaderTransform
from transforms import LabelCreatorTransform
from transforms import ComposeTransform
from transforms import TransformPickerTransform
from transforms import BrightnessTransform
from transforms import ContrastTransform
from transforms import HueTransform
from transforms import SaturationTransform
from transforms import SamplePickerTransform
from transforms import HorizontalFlipTransform
from transforms import RandomTransform
from ssdutils import get_preset_by_name
from utils import load_data_source

if sys.version_info[0] < 3:
    print("This is a Python 3 program. Use Python 3 or higher.")
    sys.exit(1)


def build_sampler(overlap, trials):
    return SamplerTransform(
        sample=True,
        min_scale=0.3,
        max_scale=1.0,
        min_aspect_ratio=0.5,
        max_aspect_ratio=2.0,
        min_jaccard_overlap=overlap,
        max_trials=trials)


def build_train_transforms(preset, num_classes, sampler_trials, expand_prob):
    # Resizing
    tf_resize = ResizeTransform(
        width=preset.image_size.w,
        height=preset.image_size.h,
        algorithms=[
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4])

    # Image distortions
    tf_brightness = BrightnessTransform(delta=32)
    tf_rnd_brightness = RandomTransform(prob=0.5, transform=tf_brightness)

    tf_contrast = ContrastTransform(lower=0.5, upper=1.5)
    tf_rnd_contrast = RandomTransform(prob=0.5, transform=tf_contrast)

    tf_hue = HueTransform(delta=18)
    tf_rnd_hue = RandomTransform(prob=0.5, transform=tf_hue)

    tf_saturation = SaturationTransform(lower=0.5, upper=1.5)
    tf_rnd_saturation = RandomTransform(prob=0.5, transform=tf_saturation)

    tf_reorder_channels = ReorderChannelsTransform()
    tf_rnd_reorder_channels = RandomTransform(
        prob=0.5, transform=tf_reorder_channels)

    # Compositions of image distortions
    tf_distort_lst = [
        tf_rnd_contrast,
        tf_rnd_saturation,
        tf_rnd_hue,
        tf_rnd_contrast
    ]
    tf_distort_1 = ComposeTransform(transforms=tf_distort_lst[:-1])
    tf_distort_2 = ComposeTransform(transforms=tf_distort_lst[1:])
    tf_distort_comp = [tf_distort_1, tf_distort_2]
    tf_distort = TransformPickerTransform(transforms=tf_distort_comp)

    # Expand sample
    tf_expand = ExpandTransform(max_ratio=4.0, mean_value=[104, 117, 123])
    tf_rnd_expand = RandomTransform(prob=expand_prob, transform=tf_expand)

    # Samplers
    samplers = [
        SamplerTransform(sample=False),
        build_sampler(0.1, sampler_trials),
        build_sampler(0.3, sampler_trials),
        build_sampler(0.5, sampler_trials),
        build_sampler(0.7, sampler_trials),
        build_sampler(0.9, sampler_trials),
        build_sampler(1.0, sampler_trials)
    ]
    tf_sample_picker = SamplePickerTransform(samplers=samplers)

    # Horizontal flip
    tf_flip = HorizontalFlipTransform()
    tf_rnd_flip = RandomTransform(prob=0.5, transform=tf_flip)

    # Transform list
    transforms = [
        ImageLoaderTransform(),
        tf_rnd_brightness,
        tf_distort,
        tf_rnd_reorder_channels,
        tf_rnd_expand,
        tf_sample_picker,
        tf_rnd_flip,
        LabelCreatorTransform(preset=preset, num_classes=num_classes),
        tf_resize
    ]
    return transforms


def build_valid_transforms(preset, num_classes):
    tf_resize = ResizeTransform(
        width=preset.image_size.w,
        height=preset.image_size.h,
        algorithms=[cv2.INTER_LINEAR])
    transforms = [
        ImageLoaderTransform(),
        LabelCreatorTransform(preset=preset, num_classes=num_classes),
        tf_resize
    ]
    return transforms


def main():
    # Parse the commandline
    parser = argparse.ArgumentParser(description='Process a dataset for SSD')
    parser.add_argument('--data-source', default='pascal_voc',
                        help='data source')
    parser.add_argument('--data-dir', default='pascal-voc',
                        help='data directory')
    parser.add_argument('--expand-probability', type=float, default=0.5,
                        help='probability of running sample expander')
    parser.add_argument('--sampler-trials', type=int, default=50,
                        help='number of time a sampler tries to find a sample')
    parser.add_argument('--preset', default='vgg300',
                        choices=['vgg300', 'vgg512'])
    args = parser.parse_args()

    print('[i] Data source:          ', args.data_source)
    print('[i] Data directory:       ', args.data_dir)
    print('[i] Expand probability:   ', args.expand_probability)
    print('[i] Sampler trials:       ', args.sampler_trials)
    print('[i] Preset:               ', args.preset)

    # Load the data source
    print('[i] Configuring the data source...')
    try:
        source = load_data_source(args.data_source)
        source.load_trainval_data(args.data_dir)
        print('[i] # training samples:   ', source.num_train)
        print('[i] # validation samples: ', source.num_valid)
        print('[i] # classes:            ', source.num_classes)
    except (ImportError, AttributeError, RuntimeError) as e:
        print('[!] Unable to load data source:', str(e))
        return 1

    # Compute the training data
    preset = get_preset_by_name(args.preset)
    with open(args.data_dir+'/train-samples.pkl', 'wb') as f:
        pickle.dump(source.train_samples, f)
    with open(args.data_dir+'/valid-samples.pkl', 'wb') as f:
        pickle.dump(source.valid_samples, f)

    with open(args.data_dir+'/training-data.pkl', 'wb') as f:
        data = {
            'preset': preset,
            'num-classes': source.num_classes,
            'colors': source.colors,
            'lid2name': source.lid2name,
            'lname2id': source.lname2id,
            'train-transforms': build_train_transforms(
                preset,
                source.num_classes,
                args.sampler_trials,
                args.expand_probability),
            'valid-transforms': build_valid_transforms(
                preset,
                source.num_classes)
        }
        pickle.dump(data, f)

    return 0


if __name__ == '__main__':
    sys.exit(main())
