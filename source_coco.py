import os
from tqdm import tqdm
import cv2
import re

from utils import Label, Box, Sample, Size
from utils import rgb2bgr, abs2prop

from pycocotools.coco import COCO

# Labels
label_defs = [
    Label('person',         rgb2bgr((0,     0,   0))),
    Label('bicycle',        rgb2bgr((111,  74,   0))),
    Label('car',            rgb2bgr((81,    0,  81))),
    Label('motorcycle',     rgb2bgr((128,  64, 128))),
    Label('airplane',       rgb2bgr((244,  35, 232))),
    Label('bus',            rgb2bgr((230, 150, 140))),
    Label('train',          rgb2bgr((70,   70,  70))),
    Label('truck',          rgb2bgr((102, 102, 156))),
    Label('boat',           rgb2bgr((190, 153, 153))),
    Label('trafficlight',   rgb2bgr((150, 120,  90))),
    Label('firehydrant',    rgb2bgr((153, 153, 153))),
    Label('stopsign',       rgb2bgr((250, 170,  30))),
    Label('parkingmeter',   rgb2bgr((220, 220,   0))),
    Label('bench',          rgb2bgr((107, 142,  35))),
    Label('bird',           rgb2bgr((52,  151,  52))),
    Label('cat',            rgb2bgr((70,  130, 180))),
    Label('dog',            rgb2bgr((220,  20,  60))),
    Label('horse',          rgb2bgr((0,     0, 142))),
    Label('sheep',          rgb2bgr((0,     0, 230))),
    Label('cow',            rgb2bgr((119,  11,  32))),
    Label('elephant',       rgb2bgr((137,  92,  34))),
    Label('bear',           rgb2bgr((167,  43,   1))),
    Label('zebra',          rgb2bgr((128, 164,  28))),
    Label('giraffe',        rgb2bgr((244,   5,  32))),
    Label('backpack',       rgb2bgr((20,   50, 140))),
    Label('umbrella',       rgb2bgr((100, 100, 100))),
    Label('handbag',        rgb2bgr((43,   43,  56))),
    Label('tie',            rgb2bgr((113,  23, 214))),
    Label('suitcase',       rgb2bgr((212, 200,  75))),
    Label('frisbee',        rgb2bgr((245, 111, 191))),
    Label('skis',           rgb2bgr((213, 255,   5))),
    Label('snowboard',      rgb2bgr((255, 255,   0))),
    Label('sportsball',     rgb2bgr((125, 125, 125))),
    Label('kite',           rgb2bgr((150, 150, 150))),
    Label('baseballbat',    rgb2bgr((175, 175, 175))),
    Label('baseballglove',  rgb2bgr((200, 200, 200))),
    Label('skateboard',     rgb2bgr((225, 225, 225))),
    Label('surfboard',      rgb2bgr((250, 250, 250))),
    Label('tennisracket',   rgb2bgr((25,   25,  25))),
    Label('bottle',         rgb2bgr((50,   50,  50))),
    Label('wineglass',      rgb2bgr((75,   75,  75))),
    Label('cup',            rgb2bgr((100, 150, 200))),
    Label('fork',           rgb2bgr((150, 200, 225))),
    Label('knife',          rgb2bgr((175, 225,  25))),
    Label('spoon',          rgb2bgr((25,   75, 125))),
    Label('bowl',           rgb2bgr((40,  100, 200))),
    Label('banana',         rgb2bgr((19,   53,  15))),
    Label('apple',          rgb2bgr((50,   20,   9))),
    Label('sandwich',       rgb2bgr((3,   153,  13))),
    Label('orange',         rgb2bgr((25,  170, 230))),
    Label('broccoli',       rgb2bgr((220, 220,   0))),
    Label('carrot',         rgb2bgr((107, 255, 235))),
    Label('hotdog',         rgb2bgr((152,  51, 152))),
    Label('pizza',          rgb2bgr((70,  142,  80))),
    Label('donut',          rgb2bgr((220,  20,  60))),
    Label('cake',           rgb2bgr((0,   100,  142))),
    Label('chair',          rgb2bgr((119,  11,  32))),
    Label('couch',          rgb2bgr((111,  74,   0))),
    Label('pottedplant',    rgb2bgr((81,    0,  81))),
    Label('bed',            rgb2bgr((128,  64, 128))),
    Label('diningtable',    rgb2bgr((244,  35, 232))),
    Label('toilet',         rgb2bgr((230, 150, 140))),
    Label('tv',             rgb2bgr((70,   70,  70))),
    Label('laptop',         rgb2bgr((102, 102, 156))),
    Label('mouse',          rgb2bgr((190, 153, 153))),
    Label('remote',         rgb2bgr((150, 120,  90))),
    Label('keyboard',       rgb2bgr((153, 153, 153))),
    Label('cellphone',      rgb2bgr((250, 170,  30))),
    Label('microwave',      rgb2bgr((220, 220,   0))),
    Label('oven',           rgb2bgr((107, 142,  35))),
    Label('toaster',        rgb2bgr((52,  151, 152))),
    Label('sink',           rgb2bgr((170, 230,  80))),
    Label('refrigerator',   rgb2bgr((220, 200, 160))),
    Label('book',           rgb2bgr((125,  25, 225))),
    Label('clock',          rgb2bgr((50,  100, 230))),
    Label('vase',           rgb2bgr((10,  211, 132))),
    Label('scissors',       rgb2bgr((22,  200, 250))),
    Label('teddybear',      rgb2bgr((100, 140, 142))),
    Label('hairdrier',      rgb2bgr((10,  100,  30))),
    Label('toothbrush',     rgb2bgr((19,  111, 132)))
]

label_ids = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90
]


class MSCOCOSource:
    def __init__(self):
        self.num_classes = len(label_defs)
        self.colors = {l.name: l.color for l in label_defs}
        self.lid2name = {i: l.name for i, l in enumerate(label_defs)}
        self.lname2id = {l.name: i for i, l in enumerate(label_defs)}
        self.num_train = 0
        self.num_valid = 0
        self.train_samples = []
        self.valid_samples = []

    def __build_sample_list(self, data_dir, data_type):
        """
        Build a list of samples for the dataset
        """
        # Read the annotations for the samples using COCO API
        coco = COCO("%s/annotations/%s.json" % (data_dir, data_type))
        samples = []
        img_dir = "%s/images/%s" % (data_dir, data_type)
        for img_name in tqdm(os.listdir(img_dir)):
            # Absolute path of the image
            img_path = "%s/%s" % (img_dir, img_name)

            # Get the file dimensions
            img = cv2.imread(img_path)
            img_size = Size(img.shape[1], img.shape[0])

            # Integer id of the image
            img_id = re.sub(r"^[0]*", "", img_name)
            img_id = re.sub(r"\.jpg", "", img_id)
            img_id = int(img_id)

            # Objects in the image
            objects = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))

            # Get boxes for all the objects
            boxes = []
            for obj in objects:
                # Get the properties of the box and convert them to the
                # proportional terms
                label_id = label_ids.index(obj['category_id'])
                box = obj['bbox']
                xmin = int(box[0])
                xmax = int(box[0]+box[2])
                ymin = int(box[1])
                ymax = int(box[1]+box[3])
                center, size = abs2prop(xmin, xmax, ymin, ymax, img_size)
                box = Box(self.lid2name[label_id], label_id, center, size)
                boxes.append(box)
            if not boxes:
                continue
            sample = Sample(img_path, boxes, img_size)
            samples.append(sample)
        return samples

    def load_trainval_data(self, data_dir):
        """
        Load the training and validation data
        :param data_dir: the directory where the dataset is stored
        """
        # Process the training samples
        self.train_samples = self.__build_sample_list(data_dir, 'train')

        # Process the validation samples
        self.valid_samples = self.__build_sample_list(data_dir, 'val')

        # Sanity check
        if len(self.train_samples) == 0:
            raise RuntimeError('No training samples found in ' + data_dir)

        if len(self.valid_samples) == 0:
            raise RuntimeError('No validation samples found in ' + data_dir)

        self.num_train = len(self.train_samples)
        self.num_valid = len(self.valid_samples)


def get_source():
    return MSCOCOSource()
