# YOLOv5 dataset utils and dataloaders

import glob
import hashlib
import json
import logging
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool
from pathlib import Path
from threading import Thread
from sklearn import preprocessing
import secrets

import tempfile
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import check_requirements, check_file, check_dataset, xywh2xyxy, xywhn2xyxy, xyxy2xywhn, \
    xyn2xy, segments2boxes, clean_str
from utils.torch_utils import torch_distributed_zero_first

import urllib3
from minio import Minio
# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads


# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def xyxy_to_training(xyxy, dim):
    x1, y1, x2, y2 = xyxy

    width = (x2 - x1) / dim[1]
    height = (y2 - y1) / dim[0]
    midX = (x1 + x2) / 2 / dim[1]
    midY = (y1 + y2) / 2 / dim[0]

    return [midX, midY, width, height]


def load_json(file_idx, cache_folder="./nlm_features/", save=True):
    filename = f"{cache_folder}/{file_idx}.json"
    if not os.path.exists(filename):
        if save:
            file_storage.download_document(
                f"bbox/features/{file_idx}.json",
                dest_file_location=f"{cache_folder}/{file_idx}.json",
            )
        else:
            filename = file_storage.download_document(
                f"bbox/features/{file_idx}.json",
            )

    with open(str(filename)) as f:
        data = json.load(f)
    return data


def correct_box(img, xyxy, threshold=None):
    def _correct_box(img, ax, fixed_ax_1, fixed_ax_2, pad_1):
        limit = img.shape[1]
        expanded = shrinked = ax
        while True:
            if (
                0 < expanded < limit + pad_1
                and np.count_nonzero(img[fixed_ax_1:fixed_ax_2, expanded - pad_1]) == 0
                and np.count_nonzero(img[fixed_ax_1:fixed_ax_2, expanded]) != 0
            ):
                return expanded
            if (
                0 < shrinked < limit + pad_1
                and np.count_nonzero(img[fixed_ax_1:fixed_ax_2, shrinked - pad_1]) == 0
                and np.count_nonzero(img[fixed_ax_1:fixed_ax_2, shrinked]) != 0
            ):
                return shrinked
            expanded -= 1
            shrinked += 1

            if expanded <= 0 and shrinked >= limit:
                raise ValueError("Can not find the box in the image")

    def _check_box(img, xyxy, threshold=0.1):
        y1, x1, y2, x2 = xyxy

        boxed_img = img[y1:y2, x1:x2]

        return (
            np.count_nonzero(boxed_img) / (boxed_img.shape[0] * boxed_img.shape[1])
            > threshold
        )

    if len(img.shape) == 3:
        img = img[:, :, 0]

    assert len(img.shape) == 2

    x1, y1, x2, y2 = [int(x) for x in xyxy]

    x1 = _correct_box(img, x1, y1, y2, 1)

    x2 = _correct_box(img, x2, y1, y2, -1)

    y1 = _correct_box(img.transpose(1, 0), y1, x1, x2, 1)

    y2 = _correct_box(img.transpose(1, 0), y2, x1, x2, -1)

    if threshold:
        valided = _check_box(img, [y1, x1, y2, x2])

    return x1, y1, x2, y2

def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha512(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    From https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT,
                  3: Image.ROTATE_180,
                  4: Image.FLIP_TOP_BOTTOM,
                  5: Image.TRANSPOSE,
                  6: Image.ROTATE_270,
                  7: Image.TRANSVERSE,
                  8: Image.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix=''):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class FileStorage:
    def __init__(
        self,
        url=None,
        access_key=None,
        secret_key=None,
        bucket="doc-store-dev",
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        url = url or os.getenv("MINIO_URL", "localhost:9000")
        httpClient = urllib3.PoolManager(maxsize=1000)

        self.minioClient = Minio(
            url,
            access_key=access_key or os.getenv("MINIO_ACCESS", "user"),
            secret_key=secret_key or os.getenv("MINIO_SECRET", "password"),
            secure=False,
            http_client=httpClient,
        )
        self.bucket = bucket
        bucket_names = {item.name for item in self.minioClient.list_buckets()}
        if bucket not in bucket_names:
            self.logger.info(f"Bucket {bucket} does not exist!")
            self.minioClient.make_bucket(bucket)

    def document_exists(self, document_location):
        document_location = document_location.replace(f"gs://{self.bucket}/", "")
        try:
            self.minioClient.stat_object(self.bucket, document_location)
        except Exception:
            return False
        return True


    def download_document(self, document_location, dest_file_location=None):
        document_location = document_location.replace(f"gs://{self.bucket}/", "")
        if self.document_exists(document_location):
            if dest_file_location is None:
                dest_file_location_handler, dest_file_location = tempfile.mkstemp()
                os.close(dest_file_location_handler)

            self.logger.info(
                f"Download document from {document_location} to {dest_file_location}",
            )
            self.minioClient.fget_object(
                self.bucket,
                document_location,
                dest_file_location,
            )
        else:
            raise Exception(
                f"Failed to download document from {document_location}. Document does not exists",
            )

        return dest_file_location


file_storage = FileStorage()
class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    def __init__(self, img_size=1344, scaler=None, doc_id=""):
        self.img_size = img_size
        self.scaler = scaler
        self.mode = "image"
        self.cap = None
        self.dimensions = 0
        self.imgs = []
        self.img_files = []
        self.page_idxs = []

        # # file_idxs = db["bboxes"].distinct("file_idx", {"block_type": "table", "audited": True})
        # if doc_id:
        #     print("loading pages for doc", doc_id)
        #     file_idxs = [doc_id]
        # else:
        #     file_idxs = db["document"].distinct("id", {"workspace_id": "704dc2bc"})
        
        # try:
        data = load_json(doc_id, save=False)
        # except Exception:
        #     print(f"error loading json for file {doc_id}")

        # self.img_names.append(str(filename.name).split(".json")[0])
        if not self.dimensions:
            self.dimensions = len(data["metadata"]["features"]) + 1
        else:
            assert self.dimensions == len(data["metadata"]["features"]) + 1

        # pages = db["bboxes"].distinct(
        #     "page_idx",
        #     {"file_idx": file_idx, "audited": False}, # "block_type": "table", 
        # )
        
        for page_idx in range(len(data["data"])):
            tokens = data["data"][page_idx]

            # HWC.
            # We hard code the padding to 0, thus image must in square (H:1344,W:1344 by default)
            features = np.zeros(
                (self.img_size, self.img_size, self.dimensions),
                dtype=np.float32,
            )

            # make features, HWF
            for token in tokens:
                # x1, y1, x2, y2 = xyxy_to_training(token["position"]["xyxy"])
                x1, y1, x2, y2 = [round(x) for x in token["position"]["xyxy"]]

                # token position mask
                features[y1:y2, x1:x2, 0] = 1

                for i, feature in enumerate(token["features"]):
                    features[y1:y2, x1:x2, i + 1] = feature
            
            self.page_idxs.append(page_idx+1)
            self.imgs.append(features)

            self.img_files.append(f"{doc_id}_{page_idx+1}.jpg")

            # print(f"features loaded for document {doc_id}, page {page_idx+1}")

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == len(self.imgs):
            raise StopIteration

        # Read image
        self.count += 1
        img = self.imgs[self.count - 1]
        page_idx = self.page_idxs[self.count - 1]

        img = self.scaler.transform(img.reshape(-1,12)).reshape(1344,1344,12)
        # NLM features in BHWC, yolo image in BCHW
        # # HWC => CWH
        # img0 = img.transpose(2, 0, 1)

        # use top-3 channel as image
        img0 = img[:, :, :3] * 255

        img = img.transpose(2, 0, 1)
        # # scale to 0-255
        # img0[img0 < 0] = 0
        # img0[img0 > 255] = 255

        # img0 = img0.astype(np.uint8)

        path = self.img_files[self.count - 1]
        s = f"image {self.count}/{len(self.imgs)} {path}: "

        # BHWC => BCWH
        # x = torch.from_numpy(img)
        # x = x.transpose(0, 2)
        # # BCWH => BCHW
        # x = x.transpose(1, 2)

        # print(x[1].shape)
        # normalize to 0-1
        # img = x * 255
        img = img * 255
        # print("shape", img.shape, img0.shape)
        return path, img, img0, page_idx, self.cap, s

# def save_inference_doc(doc_id, pages):
#     db["ml_bbox"].delete_many({"file_idx": doc_id})
#     db["ml_bbox"].insert_many(pages)
#     # db["ml_bbox"].update_many({"file_idx": doc_id}, pages, upsert=True)
     


class LoadWebcam:  # for inference
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        print(f'webcam {self.count}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            if 'youtube.com/' in s or 'youtu.be/' in s:  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f" success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                self.imgs[i] = im if success else self.imgs[i] * 0
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(
        self,
        split,
        img_size=1344,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="",
    ):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights

        self.stride = stride
        self.split = split

        self.imgs = []
        self.img_files = []
        self.labels = []
        self.dimensions = 0

        self.labels_map = {
            "table": 0,
            # "header": 1,
            # "para": 2,
        }

        file_idxs = db["bboxes"].distinct(
            "file_idx",
            {"audited": True, "block_type": {"$in": list(self.labels_map.keys())}},
        )

        for file_idx in file_idxs:
            try:
                data = load_json(file_idx)
            except:
                print(f"error loading {file_idx}")
                continue

            # self.img_names.append(str(filename.name).split(".json")[0])
            if not self.dimensions:
                self.dimensions = len(data["metadata"]["features"]) + 1
            else:
                assert self.dimensions == len(data["metadata"]["features"]) + 1

            pages = db["bboxes"].distinct(
                "page_idx",
                {
                    "file_idx": file_idx,
                    "audited": True,
                    "block_type": {"$in": list(self.labels_map.keys())},
                },
            )
            for page_idx in pages:
                if self.split and isinstance(self.split, int):
                    h = xxh32_intdigest(f"{file_idx}-{page_idx}")
                    if self.split > 5:
                        if h % 10 >= self.split:
                            print("skipping current page for training")
                            continue
                    else:
                        if h % 10 < (10 - self.split):
                            print("skipping current page for testing")
                            continue

                tokens = data["data"][page_idx]

                # HWC.
                # We hard code the padding to 0, thus image must in square (H:1344,W:1344 by default)
                features = np.zeros(
                    (self.img_size, self.img_size, self.dimensions),
                    dtype=np.float32,
                )

                # make features, HWF
                for token in tokens:
                    # x1, y1, x2, y2 = xyxy_to_training(token["position"]["xyxy"])
                    x1, y1, x2, y2 = [round(x) for x in token["position"]["xyxy"]]

                    # token position mask
                    features[y1:y2, x1:x2, 0] = 1

                    for i, feature in enumerate(token["features"]):
                        features[y1:y2, x1:x2, i + 1] = feature

                # make labels
                page_labels = []
                for label in db["bboxes"].find(
                    {
                        "file_idx": file_idx,
                        "page_idx": page_idx,
                        "audited": True,
                        "block_type": {"$in": list(self.labels_map.keys())},
                    }
                ):
                    label_type = self.labels_map[label["block_type"]]

                    try:
                        xyxy = correct_box(features, label["bbox"])
                    except Exception:
                        continue

                    label_coords = xyxy_to_training(
                        xyxy, dim=(self.img_size, self.img_size)
                    )

                    labeled = [label_type] + label_coords

                    page_labels.append(np.array(labeled))

                if page_labels:
                    self.imgs.append(features)

                    self.img_files.append(f"file:{file_idx} page:{page_idx+1}")

                    self.labels.append(page_labels)
                    print(
                        f"{len(page_labels)} labels for document {file_idx}, page {page_idx+1}"
                    )

        print(
            f"Found {len(self.imgs)} images with {sum([len(x) for x in self.labels])} labels"
        )

        # label found, label missing, label empty, label corrupted, total label.
        self.shapes = np.array(
            [[x.shape[0], x.shape[1]] for x in self.imgs], dtype=np.float64
        )
        # convert to np array
        self.labels = [np.array(x) for x in self.labels]
        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(np.array(self.imgs).reshape(-1,12))
        
        # self.scaler.fit(self.imgs)
        n = len(self.imgs)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
        img = self.imgs[index]
        # img = self.scaler.transform([img])[0]
        img = self.scaler.transform(img.reshape(-1,12)).reshape(1344,1344,12)

        # create shapes for plotting
        h0, w0 = h, w = self.img_size, self.img_size

        # orignal (h,w), (scale(h,w), padding(h,w)).
        # NOTE: padding is hard coded to 0, thus we should have
        # shape = (1344, 1344), ((1, 1), (0, 0))
        shapes = (h0, w0), ((h / h0, w / w0), (0, 0))  # for COCO mAP rescaling

        labels = self.labels[index]
        nL = len(labels)  # number of labels
        labels_out = torch.zeros((nL, 6))
        labels_out[:, 1:] = torch.from_numpy(labels)

        x = torch.from_numpy(img)

        # x = x[:, :, :1]
        # x = torch.cat([x,x,x],dim=2)
        # convert nlm features to yolo channels => BHWC to BCHW
        # BHWC => BCWH
        x = x.transpose(0, 2)
        # BCWH => BCHW
        x = x.transpose(1, 2)

        # print(x[1].shape)
        # normalize to 0-1
        x = x * 255
        # print("img", x.shape)
        # print("labels_out", len(labels_out))
        # print("index", self.img_files[index].shape)
        return x, labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if secrets.randbelow(1000)/1000 < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    im = self.imgs[i]
    if im is None:  # not cached in ram
        npy = self.img_npy[i]
        if npy and npy.exists():  # load npy
            im = np.load(npy)
        else:  # read image
            path = self.img_files[i]
            im = cv2.imread(path)  # BGR
            assert im is not None, 'Image Not Found ' + path
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    else:
        return self.imgs[i], self.img_hw0[i], self.img_hw[i]  # im, hw_original, hw_resized


def load_mosaic(self, index):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + [secrets.choice(self.indices) for _ in range(3)] # random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4


def load_mosaic9(self, index):
    # loads images in a 9-mosaic

    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + [secrets.choice(self.indices) for _ in range(8)] # random.choices(self.indices, k=8)  # 8 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    img9, labels9 = random_perspective(img9, labels9, segments9,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img9, labels9


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../datasets/coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../datasets/coco128'):  # from utils.datasets import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../datasets/coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sum([list(path.rglob(f"*.{img_ext}")) for img_ext in IMG_FORMATS], [])  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = [] # random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    for _ in range(n):
        num = secrets.randbelow(1000)/1000
        if num < weights[0]:
            indices.append(0)
        elif num < weights[0] + weights[1]:
            indices.append(1)
        else:
            indices.append(2)

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path.parent / x).unlink(missing_ok=True) for x in txt]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')  # add image to txt file


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupt
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                assert f.read() == b'\xff\xd9', 'corrupted JPEG'

        # verify labels
        segments = []  # instance segments
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, 'r') as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any([len(x) > 8 for x in l]):  # is segment
                    classes = np.array([x[0] for x in l], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                    l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                l = np.array(l, dtype=np.float32)
            if len(l):
                assert l.shape[1] == 5, 'labels require 5 columns each'
                assert (l >= 0).all(), 'negative labels'
                assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
            else:
                ne = 1  # label empty
                l = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, 5), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, ''
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False, profile=False, hub=False):
    """ Return dataset statistics dictionary with images and instances counts per split per class
    To run in parent directory: export PYTHONPATH="$PWD/yolov5"
    Usage1: from utils.datasets import *; dataset_stats('coco128.yaml', autodownload=True)
    Usage2: from utils.datasets import *; dataset_stats('../datasets/coco128_with_yaml.zip')
    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
        verbose:        Print stats dictionary
    """

    def round_labels(labels):
        # Update labels to integer class and 6 decimal place floats
        return [[int(c), *[round(x, 4) for x in points]] for c, *points in labels]

    def unzip(path):
        # Unzip data.zip TODO: CONSTRAINT: path/to/abc.zip MUST unzip to 'path/to/abc/'
        if str(path).endswith('.zip'):  # path is data.zip
            assert Path(path).is_file(), f'Error unzipping {path}, file not found'
            assert os.system(f'unzip -q {path} -d {path.parent}') == 0, f'Error unzipping {path}'
            dir = path.with_suffix('')  # dataset directory
            return True, str(dir), next(dir.rglob('*.yaml'))  # zipped, data_dir, yaml_path
        else:  # path is data.yaml
            return False, None, path

    def hub_ops(f, max_dim=1920):
        # HUB ops for 1 image 'f'
        im = Image.open(f)
        r = max_dim / max(im.height, im.width)  # ratio
        if r < 1.0:  # image too large
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(im_dir / Path(f).name, quality=75)  # save

    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_file(yaml_path), encoding='ascii', errors='ignore') as f:
        data = yaml.safe_load(f)  # data dict
        if zipped:
            data['path'] = data_dir  # TODO: should this be dir.resolve()?
    check_dataset(data, autodownload)  # download dataset if missing
    hub_dir = Path(data['path'] + ('-hub' if hub else ''))
    stats = {'nc': data['nc'], 'names': data['names']}  # statistics dictionary
    for split in 'train', 'val', 'test':
        if data.get(split) is None:
            stats[split] = None  # i.e. no test set
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])  # load dataset
        for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data['nc']))
        x = np.array(x)  # shape(128x80)
        stats[split] = {'instance_stats': {'total': int(x.sum()), 'per_class': x.sum(0).tolist()},
                        'image_stats': {'total': dataset.n, 'unlabelled': int(np.all(x == 0, 1).sum()),
                                        'per_class': (x > 0).sum(0).tolist()},
                        'labels': [{str(Path(k).name): round_labels(v.tolist())} for k, v in
                                   zip(dataset.img_files, dataset.labels)]}

        if hub:
            im_dir = hub_dir / 'images'
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(hub_ops, dataset.img_files), total=dataset.n, desc='HUB Ops'):
                pass

    # Profile
    stats_path = hub_dir / 'stats.json'
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix('.npy')
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(f'stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

            file = stats_path.with_suffix('.json')
            t1 = time.time()
            with open(file, 'w') as f:
                json.dump(stats, f)  # save stats *.json
            t2 = time.time()
            with open(file, 'r') as f:
                x = json.load(f)  # load hyps dict
            print(f'stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

    # Save, print and return
    if hub:
        print(f'Saving {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(stats, f)  # save stats.json
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats
