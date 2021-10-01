# web app setup
from flask import Flask, request, render_template, send_file, jsonify
import base64
# torch
import torch, torchvision
# Some basic setup: Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import os, json, cv2, random
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
# Set path
folder_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # app folder
template_dir = os.path.join(folder_path, "html", "templates")
# static_dir = os.path.join(folder_path, "html", "static")
model_dir = os.path.join(folder_path, "20210910")
dataset_dir = os.path.join(folder_path, "dataset")

app = Flask(__name__, template_folder=template_dir, static_url_path='/static') # , static_folder=static_dir

# setup dataset format function
def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            # assert not anno["region_attributes"]
            category_id = int(list(anno["region_attributes"].values())[0])
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": category_id,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


# prepare predictor function
def prepare_predictor(model_dir):
    # model configurations
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 30000    # 10000 iterations
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 14  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.MODEL.DEVICE='cpu' # set to cpu on MacBook

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=False)

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 #0.5   # set a custom testing threshold
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7 #0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.92 #0.5

    cfg.MODEL.RPN.NMS_THRESH = 0.5 #0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3 #0.3
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.3 #0.3

    predictor = DefaultPredictor(cfg)

    return predictor


# AI inference
def maskrcnn(im, scale = 0.8):
    """
    Input: im is cv2 color image array (h,w,3)
    Output: buffer is streaming data array
    """
    # resize if necessary (1920,1080)
    if im.shape[0] == 1080 or im.shape[1] == 1920:
        print("original size:", im.shape)
        # crop to (1080,1240)
        im = im[:, 240:1680]
        # resize to (480,640)
        im = cv2.resize(im,(640,480),interpolation=cv2.INTER_NEAREST)
        print("resized size:", im.shape)
    # resize if necessary
    elif im.shape[0] != 480 or im.shape[1] !=640:
        print("original size:", im.shape)
        # resize to (480,640)
        im = cv2.resize(im,(640,480),interpolation=cv2.INTER_NEAREST)
        print("resized size:", im.shape)

    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    # draw predicted result
    v = Visualizer(im[:, :, ::-1],
                metadata=balloon_metadata,
                scale=scale, # 0.5
                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # convert image to streaming data
    is_success, buffer = cv2.imencode(".png", out.get_image()[:, :, ::-1])
    return buffer


@app.route('/tvgh_api', methods=['POST'])
def run_tvgh_api():
    json_list = []
    if request.method == 'POST':
        if 'filename' not in request.files:
            json_list = [{'image': 'there is no filename in form!'}]
        else:
            img = request.files['filename']
            # convert to np array, img.read() is image string
            npimg = np.fromstring(img.read(), np.uint8)
            # convert to cv2 color image (h,w,3)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            # AI inference, get streaming data array
            img = maskrcnn(img, scale=1)
            # encode to base64 image string
            img_string = base64.b64encode(img)
            json_list = [{'image': img_string.decode('utf-8')}]
    return jsonify(json_list)

@app.route('/', methods=['GET'])
def run_app():
    return render_template('index.html')

if __name__ == "__main__":
    # register dataset
    for d in ["train", "val"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts(os.path.join(dataset_dir, d)))
        MetadataCatalog.get("balloon_" + d).set(thing_classes=['common_carotid_artery', 'esophagus', 'foreign_material', 'inferior_thyroid_artery', 'inferior_thyroid_vein', 'instrument', 'internal_jugular_vein', 'medical_material', 'organ_tissue', 'recurrent_laryngeal_nerve', 'scalpel', 'surgical_wound', 'thyroid', 'trachea']) #, 'background'
    balloon_metadata = MetadataCatalog.get("balloon_train")

    # load predictor
    predictor = prepare_predictor(model_dir)
    print("###############")
    print("model loaded!!!")
    print("###############")

    # app.run(debug=True, host='0.0.0.0', port=15001, ssl_context='adhoc')
    # app.run(debug=True, host='0.0.0.0', port=15001, ssl_context=(os.path.join(folder_path, "html/ssl/openaifab.com/fullchain3.pem"), os.path.join(folder_path, "html/ssl/openaifab.com/privkey3.pem")))
    app.run(debug=True, host='0.0.0.0', port=15001)
