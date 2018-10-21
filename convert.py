import os
import torch
from torch.autograd import Variable
import numpy as np
import cv2
from ssd import build_ssd
from data import VOC_CLASSES as labels
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def generate_frame(source, destination, net):
    for fname in os.listdir(source):
        if fname in destination:
            continue
        image = cv2.imread(os.path.join(source, fname))
        rgb_image = image.copy()

        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)

        xx = Variable(x.unsqueeze(0))

        xx = xx.cuda()
        y = net(xx)

        detections = y.data

        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.17:
                score = detections[0, i, j, 0]
                label_name = labels[i - 1]
                if label_name == 'person':
                    display_txt = '%s: %.2f' % (label_name, score)
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    w_tbox = abs(pt[0]-pt[2]-2)
                    h_tbox = 50 * 0.005 * w_tbox

                    color = (50, 50, 242)
                    rgb_image = cv2.rectangle(rgb_image, (pt[0], pt[1]), (pt[2], pt[3]), color, 2)
                    rgb_image = cv2.rectangle(rgb_image, (int(pt[0]-1), pt[1]), (int(pt[2]+1), int(pt[1]-h_tbox)),
                                              color, -1)
                    rgb_image = cv2.putText(rgb_image, display_txt, (int(pt[0]-1), int(pt[1]-h_tbox/3)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.005*abs(pt[0]-pt[2]),
                                            (255, 255, 255), 2)
                j += 1
        cv2.imwrite(os.path.join(destination, fname), rgb_image)


if __name__ == '__main__':
    SOURCE = '/media/federico/Volume1/remote/datasets/jjta/jta_out/prova8'
    DESTINATION = '/media/federico/Volume1/remote/datasets/jjta/jta_out/prova7'

    net = build_ssd('test', 300, 21)  # initialize SSD
    net.load_weights('./weights/ssd300_mAP_77.43_v2.pth')
    net = net.cuda()

    generate_frame(SOURCE, DESTINATION, net)
