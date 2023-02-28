import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import RRDBNet_arch as arch

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'LR'


base_dir = os.path.join(os.getcwd(), test_img_folder)
car_brands = os.listdir(base_dir)
print(car_brands)


model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

for brand in car_brands:
    path = os.path.join(base_dir, brand)
    output_dir = os.path.join(os.getcwd(), 'results', brand + '_high')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    i = 0
    for img_path in tqdm(os.listdir(path)):
        i += 1
        img = cv2.imread(os.path.join(path, img_path), cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite(os.path.join(output_dir, brand + str(i)+ '.jpg'), output)
