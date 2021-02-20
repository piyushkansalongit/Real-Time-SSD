import argparse
import pickle
import math
import sys
import cv2
import os

import tensorflow.compat.v1 as tf
import numpy as np

from average_precision import APCalculator, APs2mAP
from pascal_summary import PascalSummary
from ssdutils import get_anchors_for_preset, decode_boxes, suppress_overlaps
from ssdvgg import SSDVGG
from utils import str2bool, draw_box
from tqdm import tqdm

if sys.version_info[0] < 3:
    print("This is a Python 3 program. Use Python 3 or higher.")
    sys.exit(1)


def sample_generator(samples, image_size, batch_size):
    image_size = (image_size.w, image_size.h)
    for offset in range(0, len(samples), batch_size):
        files = samples[offset:offset+batch_size]
        images = []
        idxs = []
        for i, image_file in enumerate(files):
            image = cv2.resize(cv2.imread(image_file), image_size)
            images.append(image.astype(np.float32))
            idxs.append(offset+i)
        yield np.array(images), idxs


def main():
    # Parse commandline
    parser = argparse.ArgumentParser(description='SSD inference')
    parser.add_argument("files", nargs="*")

    parser.add_argument('--checkpoint-dir', default='pascal-voc/checkpoints', help='project name')
    parser.add_argument('--checkpoint', type=int, default=-1, help='checkpoint to restore; -1 is the most recent')

    parser.add_argument('--data-source', default="pascal-voc", help='Use test files from the data source')
    parser.add_argument('--data-dir', default='pascal-voc', help='Use test files from the data source')
    parser.add_argument('--training-data', default='pascal-voc/training-data.pkl', help='Information about parameters used for training')
    parser.add_argument('--output-dir', default='pascal-voc/annotated/train', help='directory for the resulting images')
    parser.add_argument('--annotate', type=str2bool, default='False', help="Annotate the data samples")
    parser.add_argument('--dump-predictions', type=str2bool, default='False', help="Dump raw predictions")
    parser.add_argument('--summary', type=str2bool, default='True', help='dump the detections in Pascal VOC format')
    parser.add_argument('--compute-stats', type=str2bool, default='True', help="Compute the mAP stats")
    parser.add_argument('--sample', default='train', choices=['train', 'valid'])

    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--threshold', type=float, default=0.5, help='confidence threshold')

    args = parser.parse_args()

    # Print parameters
    print('[i] Checkpoint directory: ', args.checkpoint_dir)

    print('[i] Data source:          ', args.data_source)
    print('[i] Data directory:       ', args.data_dir)
    print('[i] Training data:        ', args.training_data)
    print('[i] Output directory:     ', args.output_dir)
    print('[i] Annotate:             ', args.annotate)
    print('[i] Dump predictions:     ', args.dump_predictions)
    print('[i] Summary:              ', args.summary)
    print('[i] Compute state:        ', args.compute_stats)
    print('[i] Sample:               ', args.sample)

    print('[i] Batch size:           ', args.batch_size)
    print('[i] Threshold:            ', args.threshold)

    # Check if we can get the checkpoint
    state = tf.train.get_checkpoint_state(args.checkpoint_dir)
    if state is None:
        print('[!] No network state found in ' + args.checkpoint_dir)
        return 1

    try:
        checkpoint_file = state.all_model_checkpoint_paths[args.checkpoint]
    except IndexError:
        print('[!] Cannot find checkpoint ' + str(args.checkpoint_file))
        return 1

    metagraph_file = checkpoint_file + '.meta'

    if not os.path.exists(metagraph_file):
        print('[!] Cannot find metagraph ' + metagraph_file)
        return 1

    # Load the training data parameters
    try:
        with open(args.training_data, 'rb') as f:
            data = pickle.load(f)
        preset = data['preset']
        colors = data['colors']
        lid2name = data['lid2name']
        image_size = preset.image_size
        anchors = get_anchors_for_preset(preset)
    except (FileNotFoundError, IOError, KeyError) as e:
        print('[!] Unable to load training data:', str(e))
        return 1

    # Load the samples according to data source and sample type
    try:
        if args.sample == 'train':
            with open(args.data_dir+'/train-samples.pkl', 'rb') as f:
                samples = pickle.load(f)
        else:
            with open(args.data_dir+'/valid-samples.pkl', 'rb') as f:
                samples = pickle.load(f)
        num_samples = len(samples)
        print('[i] # samples:         ', num_samples)
    except (ImportError, AttributeError, RuntimeError) as e:
        print('[!] Unable to load data source:', str(e))
        return 1

    # Create a list of files to analyse and make sure that the output directory exists
    files = []

    for sample in samples:
        files.append(sample.filename)

    files = list(filter(lambda x: os.path.exists(x), files))
    if files:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    # Print model and dataset stats
    print('[i] Network checkpoint:', checkpoint_file)
    print('[i] Metagraph file:    ', metagraph_file)
    print('[i] Image size:        ', image_size)
    print('[i] Number of files:   ', len(files))

    # Create the network
    if args.compute_stats:
        ap_calc = APCalculator()

    if args.summary:
        summary = PascalSummary()

    with tf.Session() as sess:
        print('[i] Creating the model...')
        net = SSDVGG(sess, preset)
        net.build_from_metagraph(metagraph_file, checkpoint_file)

        # Process the images
        generator = sample_generator(files, image_size, args.batch_size)
        n_sample_batches = int(math.ceil(len(files)/args.batch_size))
        description = '[i] Processing samples'

        for x, idxs in tqdm(generator, total=n_sample_batches, desc=description, unit='batches'):
            feed = {net.image_input: x, net.keep_prob: 1}
            enc_boxes = sess.run(net.result, feed_dict=feed)

            # Process the predictions
            for i in range(enc_boxes.shape[0]):
                boxes = decode_boxes(enc_boxes[i], anchors, args.threshold, lid2name, None)
                boxes = suppress_overlaps(boxes)[:200]
                filename = files[idxs[i]]
                basename = os.path.basename(filename)

                # Annotate samples
                if args.annotate:
                    img = cv2.imread(filename)
                    for box in boxes:
                        draw_box(img, box[1], colors[box[1].label])
                    fn = args.output_dir+'/images/'+basename
                    cv2.imwrite(fn, img)

                # Dump the predictions
                if args.dump_predictions:
                    raw_fn = args.output_dir+'/'+basename+'.npy'
                    np.save(raw_fn, enc_boxes[i])

                # Add predictions to the stats calculator and to the summary
                if args.compute_stats:
                    ap_calc.add_detections(samples[idxs[i]].boxes, boxes)

                if args.summary:
                    summary.add_detections(filename, boxes)

    # Compute and print the stats
    if args.compute_stats:
        aps = ap_calc.compute_aps()
        for k, v in aps.items():
            print('[i] AP [{0}]: {1:.3f}'.format(k, v))
        print('[i] mAP: {0:.3f}'.format(APs2mAP(aps)))

    # Write the summary files
    if args.summary:
        summary.write_summary(args.output_dir+"/summaries")

    print('[i] All done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
