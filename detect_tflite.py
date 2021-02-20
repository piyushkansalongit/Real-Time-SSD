import tensorflow as tf
import argparse
import pickle
import numpy as np
import sys
import cv2
import json
import os

from ssdutils import get_anchors_for_preset, decode_boxes, suppress_overlaps
from utils import draw_box
from tqdm import tqdm

if sys.version_info[0] < 3:
    print("This is a Python 3 program. Use Python 3 or higher.")
    sys.exit(1)


# Start the show
def main():
    # Parse the commandline
    parser = argparse.ArgumentParser(description='SSD inference')
    parser.add_argument('--model', default='./pascal-voc/models/e225-SSD300-VGG16-PASCALVOC.tflite', help='model file')
    parser.add_argument('--training-data', default='./pascal-voc/training-data.pkl', help='training data')
    parser.add_argument("--input-dir", default='./test/in', help='input directory')
    parser.add_argument('--output-dir', default='./test/out', help='output directory')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    args = parser.parse_args()

    # Print parameters
    print('[i] Model:         ', args.model)
    print('[i] Training data: ', args.training_data)
    print('[i] Input dir:     ', args.input_dir)
    print('[i] Output dir:    ', args.output_dir)
    print('[i] Batch size:    ', args.batch_size)

    # Load the training data
    with open(args.training_data, 'rb') as f:
        data = pickle.load(f)
        preset = data['preset']
        colors = data['colors']
        lid2name = data['lid2name']
        anchors = get_anchors_for_preset(preset)

    # Get the input images
    images = os.listdir(args.input_dir)
    images = ["%s/%s" % (args.input_dir, image) for image in images]

    # Create the output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    # Run the detections in batches
    for i in tqdm(range(0, len(images), args.batch_size)):
        batch_names = images[i:i+args.batch_size]
        batch_imgs = []
        batch = []
        for f in batch_names:
            img = cv2.imread(f)
            batch_imgs.append(img)
            img = cv2.resize(img, (300, 300))
            batch.append(img.astype(np.float32))

        batch = np.array(batch)

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        interpreter.set_tensor(input_details[0]['index'], batch)
        interpreter.invoke()

        output_details = interpreter.get_output_details()
        enc_boxes = interpreter.get_tensor(output_details[0]['index'])

        for i in range(len(batch_names)):
            boxes = decode_boxes(enc_boxes[i], anchors, 0.5, lid2name, None)
            boxes = suppress_overlaps(boxes)[:200]
            name = os.path.basename(batch_names[i])
            meta = {}
            for j, box in enumerate(boxes):
                draw_box(batch_imgs[i], box[1], colors[box[1].label])
                box_data = {}
                box_data['Label'] = box[1].label,
                box_data['LabelID'] = str(box[1].labelid)
                box_data['Center'] = [box[1].center.x, box[1].center.y]
                box_data['Size'] = [box[1].size.w, box[1].size.h]
                box_data['Confidence'] = str(box[0])
                meta["prediction_%s" % (j+1)] = box_data
            with open(os.path.join(args.output_dir, name+'.json'), 'w') as f:
                json.dump(meta, f, indent=4)

            cv2.imwrite(os.path.join(args.output_dir, name), batch_imgs[i])


if __name__ == '__main__':
    main()
