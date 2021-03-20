import cv2
import torch, time, base64
import numpy as np
from pathlib import Path

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

CONF_THRESH = 0.25
IOU_THRESH = 0.45
WORK_JPG = '/tmp/in.png'
SAVE_PATH = '/tmp/out.png'

def handler(event, context):
    device = select_device('cpu')
    
    # load_model
    #model =attempt_load('yolov5s.pt', map_location=device)
    model =attempt_load(['pre_best.pt','v5s_exp3.pt','v5x_exp6.pt'], map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(640, s=stride)
    
    img_bin = base64.b64decode(event['img'].encode('utf-8'))
    img_array = np.frombuffer(img_bin,dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    cv2.imwrite(WORK_JPG,img)
    dataset = LoadImages(WORK_JPG, img_size=imgsz, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]
    
    t0 = time.time()
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=True)[0]

        # Apply NMS
        pred = non_max_suppression(pred, CONF_THRESH, IOU_THRESH, classes=None, agnostic=False)
        t2 = time_synchronized()
    
        # Process detections

        labels = []
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    
                    # Write to file
                    '''
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    '''

                    label = f'{names[int(cls)]} {conf:.2f}'
                    label_test = f'{names[int(cls)]}'
                    labels.append(label_test) # label store 
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
        
        print(f'{s}Done. ({t2 - t1:.3f}s)') 
        cv2.imwrite(SAVE_PATH, im0)
        
    with open(SAVE_PATH,'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    print("done")
    
    # label exchange
    l_names = ['N','R','SR','UR',
               'Usagi','Neko','Kuma','Inu',
               'Hiyoko','Penguin','Panda','Hamster','Tanuki','Hitsuzi','Tako','Koara',
               'Tenin-san','Unknown','Girl']
    L = []
    for l in labels:
        if l == 'Usagi':  L.append('ウサギ')
        if l == 'Neko': L.append('ネコ') 
        if l == 'Kuma': L.append('クマ') 
        if l == 'Inu': L.append('イヌ') 
        if l == 'Hiyoko': L.append('ヒヨコ') 
        if l == 'Penguin': L.append('ペンギン') 
        if l == 'Panda': L.append('パンダ') 
        if l == 'Hamster': L.append('ハムスター') 
        if l == 'Tanuki': L.append('タヌキ') 
        if l == 'Hitsuzi': L.append('ヒツジ') 
        if l == 'Tako': L.append('タコ') 
        if l == 'Koara': L.append('コアラ') 
        if l == 'Tenin-san': L.append('店員さん') 
        if l == 'Unknown': L.append('かぶりもの') 
        if l == 'Girl': L.append('ギャル') 
        if l == 'N': L.append('N') 
        if l == 'R': L.append('R') 
        if l == 'SR': L.append('SR') 
        if l == 'UR': L.append('UR') 
    # rm multi label
    L = list(set(L))


    response = {'img':img_b64, 'label':L}
    return response

if __name__ == '__main__':
    input_file = './data/images/20.jpg'
    data = {}
    with open(input_file,'rb') as f:
        data['img']= base64.b64encode(f.read()).decode('utf-8')
    output_b64 = handler(data,'context')
    img_bin = base64.b64decode(output_b64['img'].encode('utf-8'))
    img_array = np.frombuffer(img_bin,dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    cv2.imwrite('./out.png',img)
    
    output_label = output_b64['label']
    print(output_label)

    exit()