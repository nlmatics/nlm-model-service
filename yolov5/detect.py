import sys
import time
import pickle
from pathlib import Path
import hashlib
import torch
import logging

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
# if FILE.parents[1].as_posix() in sys.path:  # remove /nlm-model-server from path
#     sys.path.remove(FILE.parents[1].as_posix())

from flask_jsonpify import jsonify
from flask import make_response
from flask_restful import Resource
from flask_restful import reqparse

from yolo_models.experimental import attempt_load
from utils.datasets import LoadImages, get_hash
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync

# logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
        

def load_model(weights_path, scaler_path):
    w = weights_path
    scaler_path = w[:w.rfind("/")+1]+"scaler.pickle" if not scaler_path else scaler_path
    # Dataloader
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    versions = {
        "weights": get_hash(weights_path),
        "scaler": get_hash(scaler_path),
    }

    device = select_device('')
    # Load model
    # stride, self.names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    model = attempt_load(weights_path, map_location=device)  # load FP32 model
    # stride = int(self.model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    # self.imgsz = check_img_size(imgsz, s=stride)  # check image size

    return model, scaler, device, names, versions
    

class Inference(Resource):
    def __init__(self,
        model, 
        scaler,
        device,
        names,
        versions={},
        imgsz=1344,  # inference size (pixels)
        ):

        # Initialize
        self.model = model
        self.scaler = scaler 
        self.device = device
        self.names = names
        self.imgsz = imgsz
        self.versions = versions
        
        # super().__init__()
        self.req_parser = reqparse.RequestParser()
        self.req_parser.add_argument(
            "doc_id",
            type=str,
        )

    @torch.no_grad()
    def post(self,
        conf_thres=0.10,
        iou_thres=0.45,  # NMS IOU threshold
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        max_det=1000  # maximum detections per image):
        ):

        args = self.req_parser.parse_args()
        doc_id = args["doc_id"]
        config = {
            "conf_thres": conf_thres, 
            "iou_thres": iou_thres,
            "classes": classes,
            "agnostic_nms": agnostic_nms, 
            "max_det": max_det,
            "imgsz": self.imgsz,
        }
        # load image
        dataset = LoadImages(img_size=self.imgsz, scaler=self.scaler, doc_id=doc_id)

        t0 = time.time()
        pages = []
        for path, img, im0s, page_idx, vid_cap, s in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            # Inference
            t1 = time_sync()
            pred = self.model(img, augment=False, visualize=False)[0]            
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            t2 = time_sync()
            page_labels = {}
            # Process predictions
            for i, det in enumerate(pred):  # detections per image
                p, _, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # img.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                # imc = im0  # for save_crop

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # if save_txt:  # Write to file
                        #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        #     with open(txt_path + '.txt', 'a') as f:
                        #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        c = int(cls)  # integer class
                        label_name = self.names[c]
                        # label = f'{self.names[c]} {conf:.2f}'
                        xyxy = torch.tensor(xyxy).view(1, 4).view(-1).to(dtype=torch.int).tolist()
                        if label_name in page_labels:
                            page_labels[label_name].append(xyxy)
                        else:
                            page_labels[label_name]= [xyxy]
                        # plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)



                # Print time (inference + NMS)
                # print(s)
                logger.info(f'{s}Done. ({t2 - t1:.3f}s)')

                # Save results (image with detections)
                # if save_img:
                #     cv2.imwrite(save_path, im0)
            pages.append({
                "file_idx": doc_id,
                "page_idx": page_idx,
                "labels": page_labels,
                "audited": False,
                "version": self.versions,
                "config": config
            })
        logger.info(f"Inference completed ({time_sync() - t0:.3f}s)")
        
        return make_response(jsonify(pages), 200)
