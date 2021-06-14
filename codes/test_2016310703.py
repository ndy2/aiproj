import os
import glob
import argparse
import pickle
import cv2
import numpy as np
from utils_2016310703 import *

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor']


def get_args():
    parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size", type=int, default=416, help="The common width and height for all images")
    parser.add_argument("--conf_threshold", type=float, default=0.35)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--pre_trained_model_path", type=str, default="trained_models/model_2016310703.pt")
    parser.add_argument("--input", type=str, default="data/test")
    parser.add_argument("--output", type=str, default="test_out")
    parser.add_argument("--mAP", type=bool, default=True)

    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        model = torch.load(opt.pre_trained_model_path)
    else:
        model = torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage)
    
    model.eval()
    colors = pallete()

    if not os.path.exists(opt.output):
        os.makedirs(opt.output)
    if not os.path.exists(opt.output+'/images'):
        os.makedirs(opt.output+'/images')
    if not os.path.exists(opt.output+'/yolo_text'):
        os.makedirs(opt.output+'/yolo_text')
    
    for image_path in glob.iglob(opt.input +'/test_images/'+'*.jpg'):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        image = cv2.resize(image, (opt.image_size, opt.image_size))
        image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
        image = image[None, :, :, :]
        width_ratio = float(opt.image_size) / width
        height_ratio = float(opt.image_size) / height
        data = Variable(torch.FloatTensor(image))
        if torch.cuda.is_available():
            data = data.cuda()

        with torch.no_grad():
            logits = model(data)

            predictions = post_processing(logits, opt.image_size, CLASSES, model.anchors, opt.conf_threshold,
                                          opt.nms_threshold)
        
        # print(image_path)
        idx = os.path.splitext(os.path.split(image_path)[-1])[0] 
        output_image = cv2.imread(image_path)
        outfile_path = opt.output+ '/images/' + os.path.split(image_path)[-1][:-4] +"_prediction.jpg"

        if len(predictions) != 0:
            predictions = predictions[0]
            output_text = open(opt.output+'/yolo_text/'+idx +'.txt', 'w', encoding = 'UTF8'  )
            for pred in predictions:
                xmin = int(max(pred[0] / width_ratio, 0))
                ymin = int(max(pred[1] / height_ratio, 0))
                xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
                ymax = int(min((pred[1] + pred[3]) / height_ratio, height))
                color = colors[CLASSES.index(pred[5])]
                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
                text_size = cv2.getTextSize(pred[5] + ' : %.2f' % pred[4], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
                cv2.putText(
                    output_image, pred[5] + ' : %.2f' % pred[4],
                    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)
                # print("Object: {}, Bounding box: ({},{}) ({},{})".format(pred[5], xmin, xmax, ymin, ymax))
                output_text.write("{} {} {} {} {} {}\n".format(pred[5], pred[4], xmin, xmax, ymin, ymax))
            cv2.imwrite(outfile_path, output_image)
        else :
            cv2.imwrite(outfile_path, output_image)
            output_text = open(opt.output+'/yolo_text/'+idx +'.txt', 'w', encoding = 'UTF8'  )

    if(opt.mAP):
        voc_xml_to_yolo_txt(opt.input +'/test_annotations/voc_xml', opt.input +'/test_annotations/yolo_text/')
        mAP(opt.input +'/test_annotations/yolo_text',  opt.output + '/yolo_text')



if __name__ == "__main__":
    opt = get_args()
    test(opt)
