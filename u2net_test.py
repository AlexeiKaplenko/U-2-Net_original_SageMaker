import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
import cv2

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def main():

    # --------- 1. get image path and name ---------
    model_name='u2netp'#u2netp

    image_dir = 'image_dir'
    prediction_dir = 'prediction_dir'

    # prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)

    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name+'.pth')

    img_name_list = sorted(glob.glob(image_dir + os.sep + '*'))
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=32)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir))
#     net = torch.load(model_dir)
    
    # for gpu inference    
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        hires_image = cv2.imread(img_name_list[i_test])
        hires_image = cv2.cvtColor(hires_image, cv2.COLOR_BGR2RGB)

        height, width = hires_image.shape[:2]

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        # # for gpu inference    
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        # for cpu inference    
        # inputs_test = Variable(inputs_test)

        print('inputs_test', torch.max(inputs_test))

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        #from U2Net for video
        ###########################
        prediction = d1[0,0,:,:]
        pred = np.uint8(prediction.cpu().detach().numpy() * 255)
        # pred = cv2.resize(pred, (height, width), cv2.INTER_LINEAR)
        pred = cv2.resize(pred, (width, height), cv2.INTER_CUBIC)

        #OTSU binarization
        _, binary_mask = cv2.threshold(pred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # pred = pred * (binary_mask/255)
        pred = binary_mask
        #############################

        print('pred_shape', pred.shape)

        pred = cv2.resize(pred, (width, height), cv2.INTER_CUBIC)

        inputs = inputs_test.cpu().detach().numpy()[0]
        input_image = inputs[:3] * 255

        input_image = np.moveaxis(input_image, 0, 2)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        final_image = cv2.cvtColor(hires_image, cv2.COLOR_RGB2RGBA)
        print('final_image_shape', final_image.shape)
        final_image[:,:,3] = pred
        # final_image = final_image * pred

        image_pil = Image.fromarray(np.uint8(final_image))

        # # for black background
        # black_background = Image.new("RGBA", image_pil.size, "BLACK") # Create a white rgba background
        # black_background.paste(image_pil, (0, 0), image_pil)  # Paste the image on the background. Go to the links given below for details.
        # image_pil = black_background.convert('RGB')

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)

        cv2.imwrite(os.path.join(prediction_dir, img_name_list[i_test].split(os.sep)[-1]), pred)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
