from __future__ import print_function
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings('ignore')
import time
import network
from dataset.dataset import MyDataset
from STN import AffineTransform
from loss import ComputeLoss


def show_plot(iteration, loss, name):
    plt.plot(iteration, loss)
    plt.savefig('./%s' % name)
    plt.show()


def train():
    time_train_start = time.time()
    print('Using device ' + str(device) + ' for training!')
    scale1_model = network.Scale1TPsReg().to(device)
    scale2_model = network.Scale2TPsReg().to(device)
    scale3_model = network.Scale3TPsReg().to(device)
    scale1_model.train()
    scale2_model.train()
    scale3_model.train()
    Epoch = []
    Loss_per_epoch = []
    scale_1_learn_rate = 0.00025
    scale_2_learn_rate = 0.00025
    scale_3_learn_rate = 0.00025
    for epoch in range(start_train_epoch, start_train_epoch + train_epoch):
        if (epoch - start_train_epoch) % 10 == 0:
            scale_1_learn_rate = scale_1_learn_rate * 0.87
            scale_2_learn_rate = scale_2_learn_rate * 0.87
            scale_3_learn_rate = scale_3_learn_rate * 0.87
        scale_1_optimizer = optim.SGD(scale1_model.parameters(), lr=scale_1_learn_rate)
        scale_2_optimizer = optim.SGD(scale2_model.parameters(), lr=scale_2_learn_rate)
        scale_3_optimizer = optim.SGD(scale3_model.parameters(), lr=scale_3_learn_rate)
        print('Epoch: %d' % epoch)
        Epoch.append(epoch)
        Loss_per_batchsize = []
        time_epoch_start = time.time()
        for i, data in enumerate(dataloader):
            # ref_tensor = data[0]
            # sen_tensor = data[1]
            ref_tensor = data['tensor1']
            sen_tensor = data['tensor2_warp']
            "Scale: 1"
            scale_1_optimizer.zero_grad()
            scale_1_affine_parameter = scale1_model(ref_tensor, sen_tensor)
            sen_tran_tensor, ref_inv_tensor, inv_affine_parameter_1 = AffineTransform(ref_tensor, sen_tensor,
                                                                                      scale_1_affine_parameter)
            loss_1 = ComputeLoss(ref_tensor, sen_tran_tensor, sen_tensor, ref_inv_tensor)
            "Scale: 2"
            scale_2_optimizer.zero_grad()
            scale_2_affine_parameter = scale2_model(ref_tensor, sen_tran_tensor)
            sen_tran_tensor, ref_inv_tensor, inv_affine_parameter_2 = AffineTransform(ref_tensor, sen_tran_tensor,
                                                                                      scale_2_affine_parameter)
            loss_2 = ComputeLoss(ref_tensor, sen_tran_tensor, sen_tensor, ref_inv_tensor)
            "Scale: 3"
            scale_3_optimizer.zero_grad()
            scale_3_affine_parameter = scale3_model(ref_tensor, sen_tran_tensor)
            sen_tran_tensor, ref_inv_tensor, inv_affine_parameter_3 = AffineTransform(ref_tensor, sen_tran_tensor,
                                                                                      scale_3_affine_parameter)
            # inv_affine_parameter = torch.matmul(torch.matmul(inv_affine_parameter_1, inv_affine_parameter_2),
            #                                     inv_affine_parameter_3)
            loss_3 = ComputeLoss(ref_tensor, sen_tran_tensor, sen_tensor, ref_inv_tensor)
            loss = 0.14285714 * loss_1 + 0.28571429 * loss_2 + 0.57142857 * loss_3
            pp = loss.detach().cpu()
            if not np.isnan(pp):
                loss.backward()
                Loss_per_batchsize.append(pp)
            scale_1_optimizer.step()
            scale_2_optimizer.step()
            scale_3_optimizer.step()
            if i % 50 == 0:
                print('[Epoch: %d]%f%% loss: %f' % (epoch, i / total_epoch * 100, loss))
        loss_per_epoch = np.mean(Loss_per_batchsize)
        save_loss_info = 'Epoch %d average loss is %f\n' % (epoch, loss_per_epoch)
        print(save_loss_info)
        with open(loss_info_save_path, "a") as file:
            file.write(save_loss_info)
        Loss_per_epoch.append(loss_per_epoch)
        scale_1_model_name = 'scale_1_model_' + str(epoch) + '.pth'
        sacle_1_model_save_path = os.path.join(model_save_path, 'scale_1', scale_1_model_name)
        torch.save(scale1_model, sacle_1_model_save_path)
        scale_2_model_name = 'scale_2_model_' + str(epoch) + '.pth'
        sacle_2_model_save_path = os.path.join(model_save_path, 'scale_2', scale_2_model_name)
        torch.save(scale2_model, sacle_2_model_save_path)
        scale_3_model_name = 'scale_3_model_' + str(epoch) + '.pth'
        sacle_3_model_save_path = os.path.join(model_save_path, 'scale_3', scale_3_model_name)
        torch.save(scale3_model, sacle_3_model_save_path)
        print("Epoch: {} epoch time: {:.1f}s".format(epoch, time.time() - time_epoch_start))
    show_plot(Epoch, Loss_per_epoch, 'Epoch_loss')
    print("Total train time: {:.1f}s".format(time.time() - time_train_start))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """ PATH Setting """
    # train_modal1_folder = 'E://dataset/Optical-Optical/ref'  # Folder path for training image pairs for modal-1 and model-2.
    # train_modal2_folder = 'E://dataset/Optical-Optical/sen'
    train_modal1_folder = './data/Optical-Optical/time1'
    train_modal2_folder = './data/Optical-Optical/time1'
    # Ensure that the paired images have the same name on both folders, and both folders have same number of images.
    save_model_folder = './saved/model/'  # Folder path for saved model for modal-1 and model-2.
    save_loss_folder = './saved/loss_fig/'  # Folder path for saved loss information

    data_path = ''
    # batch_size = 2
    train_epoch = 340
    start_train_epoch = 0
    loss_info_save_path = ''

    """ Training Setting """
    learn_rate = 0.0001
    weight_decay = 1e-4
    batch_size = 8
    train_epoch = 100
    image_size = 512
    num_sample = 200
    supervised_criterion = True
    unsupervised_criterion = True
    similarity = 'NCC'
    descriptor = 'defult'
    mask = True
    alpha_ = 5  # balance between supervised and unsupervised criterion, which only works on both criterions set True.

    """ DATA Geometric Distortion Setting """
    # Set the parameter range for simulating geometric distortion.
    # [a, b, c]: a refers to min, b refers to max, and c refers to interval.
    range_translation_pixel_x = [-10, 10, 1]
    range_translation_pixel_y = [-10, 10, 1]
    range_scale_x = [0.9, 1.1, 0.05]
    range_scale_y = [0.9, 1.1, 0.05]
    range_rotate_angle = [-10, 10, 2]
    range_shear_angle_x = [0, 0, 0]
    range_shear_angle_y = [0, 0, 0]
    translation_x_equals_y = False
    scale_x_equals_y = True
    shear_x_equals_y = False
    range_geo_distortion = {
        'range_translation_pixel_x': range_translation_pixel_x,
        'range_translation_pixel_y': range_translation_pixel_y,
        'range_scale_x': range_scale_x,
        'range_scale_y': range_scale_y,
        'range_rotate_angle': range_rotate_angle,
        'range_shear_angle_x': range_shear_angle_x,
        'range_shear_angle_y': range_shear_angle_y,
        'translation_x_equals_y': translation_x_equals_y,
        'scale_x_equals_y': scale_x_equals_y,
        'shear_x_equals_y': shear_x_equals_y
    }
    dataset = MyDataset(train_modal1_folder, train_modal2_folder, range_geo_distortion)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    total_epoch = len(dataloader)
    model_save_path = os.path.join(data_path, 'train', 'save_model')
    train()
