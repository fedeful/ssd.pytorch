import os
import torch
from torch.autograd import Variable
import numpy as np
import cv2
from ssd import build_ssd
from data import VOC_CLASSES as labels
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def generate_frame(source, net):
    tot_score = 0
    tot_count = 0
    images_name = [fname for fname in os.listdir(source) if fname.split('_')[0] == 'gt']
    for fname in images_name:
        if tot_count % 500 == 0:
            print(tot_count)
        image = cv2.imread(os.path.join(source, fname))

        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)

        xx = Variable(x.unsqueeze(0))

        xx = xx.cuda()
        y = net(xx)

        detections = y.data

        max_detection = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.01:
                score = detections[0, i, j, 0]
                label_name = labels[i - 1]
                if label_name == 'person' and score > max_detection:
                    max_detection = score
                j += 1
        tot_score += max_detection
        tot_count += 1
    return tot_score/tot_count


if __name__ == '__main__':
    SOURCE = '/home/federico/Desktop/pedgenlog/person_generated'

    net = build_ssd('test', 300, 21)  # initialize SSD
    net.load_weights('./weights/ssd300_mAP_77.43_v2.pth')
    net = net.cuda()

    a = generate_frame(SOURCE, net)
    with open('./sdscoregt.txt', 'w') as fo:
        fo.write('{}'.format(a))
