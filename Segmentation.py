from PIL import Image
import numpy as np
import os
import torch

test_case = 'male/0006.jpg'
result_name = "result/Joint_256_('160_normal_4', '32_sobel_4', '32_canny_4')_2021-03-20_00-50-01"

color = [(114, 128, 250),  # 鲜肉色
         (0, 165, 255),  # 橙色
         (255, 191, 0),  # 深天蓝
         (50, 205, 50)]  # 酸橙绿
classification = ['其他', '牙冠', '牙齿', '牙龈']

# load
net = torch.load(result_name + '.pth')
image_copy = Image.open('H:/DentalClassification/data/cut_' + test_case).convert('RGB')
image_normal = Image.open('H:/DentalClassification/data/cut_' + test_case).convert('L')
image_sobel = Image.open('H:/DentalClassification/data/cut_sobel_' + test_case).convert('L')
image_canny = Image.open('H:/DentalClassification/data/cut_canny_' + test_case).convert('L')

for i in range(700):
    for j in range(1460):
        x = (torch.Tensor(np.expand_dims(np.expand_dims(np.array(image_normal.crop((j-80, i-80, j+80, i+80))), axis=0), axis=0)).cuda(),
             torch.Tensor(np.expand_dims(np.expand_dims(np.array(image_sobel.crop((j-16, i-16, j+16, i+16))), axis=0), axis=0)).cuda(),
             torch.Tensor(np.expand_dims(np.expand_dims(np.array(image_canny.crop((j-16, i-16, j+16, i+16))), axis=0), axis=0)).cuda())
        y = torch.argmax(net.forward(x[0], x[1], x[2])).cpu().numpy()
        image_copy.putpixel((j, i), color[y])
        print("Painting (%d, %d) " % (i, j) + classification[y])

image_copy.save(result_name + '_Segmentation.jpg')
image_copy.show()