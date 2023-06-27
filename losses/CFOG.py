import torch
import torch.nn.functional as F


def calculate_descriptor(I):
    filter_kernel = torch.tensor(
        [[0.0029690167439505, 0.0133062098910137, 0.0219382312797146, 0.0133062098910137, 0.00296901674395050],
         [0.0133062098910137, 0.0596342954361801, 0.0983203313488458, 0.0596342954361801, 0.0133062098910137],
         [0.0219382312797146, 0.0983203313488458, 0.162102821637127, 0.0983203313488458, 0.0219382312797146],
         [0.0133062098910137, 0.0596342954361801, 0.0983203313488458, 0.0596342954361801, 0.0133062098910137],
         [0.0029690167439505, 0.0133062098910137, 0.0219382312797146, 0.0133062098910137,
          0.00296901674395050]]).unsqueeze(0).unsqueeze(0).cuda()
    displacement_x = [0, -0.707106781186548, -1, -0.707106781186548, 0, 0.707106781186548, 1, 0.707106781186548]
    displacement_y = [1, 0.707106781186548, 0, -0.707106781186548, -1, -0.707106781186548, 0, 0.707106781186548]
    xs0 = [1, -1, 0, 0]
    ys0 = [0, 0, 1, -1]
    descriptor_dim = len(xs0)
    [batch_size, channel, height, width] = I.size()
    Dp = torch.FloatTensor(batch_size, descriptor_dim, height, width).zero_().cuda()
    for i in range(descriptor_dim):
        temp = apply_displacement(I, xs0[i], ys0[i]).cuda()
        temp1 = I - temp
        temp2 = temp1.mul(temp1)
        aa = F.conv2d(input=temp2, weight=filter_kernel, padding=2)
        Dp[:, i, :, :] = aa.squeeze()
    V = Dp.mean(dim=1, keepdim=True)
    V_m = V.mean()
    val1 = 0.001 * V_m
    val2 = 1000 * V_m
    V1 = torch.min(torch.max(V, val1), val2)
    descriptor_dim1 = len(displacement_x)
    descriptor = torch.FloatTensor(batch_size, descriptor_dim1, height, width).zero_().cuda()

    for i in range(descriptor_dim1):
        temp = apply_shift(I, displacement_x[i], displacement_y[i])
        temp1 = I - temp
        temp2 = temp1.mul(temp1)
        m_temp = F.conv2d(input=temp2, weight=filter_kernel, padding=2)
        m_temp1 = -m_temp
        m_temp2 = torch.div(m_temp1, V1)
        descriptor[:, i, :, :] = m_temp2.exp().squeeze()
    max1 = descriptor.sum(dim=1, keepdim=False)
    for i in range(descriptor_dim1):
        descriptor[:, i, :, :] = torch.div(descriptor[:, i, :, :].clone(), max1)
    return descriptor


def apply_displacement(image, x, y):
    [batch_size, channel, height, width] = image.size()
    modified_image = image.clone()
    x_start = max(1, x + 1) - 1
    x_end = min(width, width + x) - 1
    y_start = max(1, y + 1) - 1
    y_end = min(height, height + y) - 1
    x_start_modified = max(1, -x + 1) - 1
    x_end_modified = min(width, width - x) - 1
    y_start_modified = max(1, -y + 1) - 1
    y_end_modified = min(height, height - y) - 1
    modified_image[:, :, y_start:y_end + 1, x_start:x_end + 1] = image[:, :, y_start_modified:y_end_modified + 1,
                                                                 x_start_modified:x_end_modified + 1]
    return modified_image


def apply_shift(input_tensor, x, y):
    [batch, channels, height, width] = input_tensor.size()
    matrix = torch.tensor([[[1, 0, -y * 2 / height], [0, 1, -x * 2 / width]]]).cuda()
    matrix = matrix.repeat(batch, 1, 1)
    grid = F.affine_grid(matrix, input_tensor.size())
    shifted_tensor = F.grid_sample(input_tensor, grid, mode='bicubic', padding_mode='border')
    return shifted_tensor


def calculate_inverse_transform(input1, input2):
    transformed_input1 = torch.fft.fftn(input1, dim=[1, 2, 3])
    transformed_input2 = torch.fft.fftn(input2, dim=[1, 2, 3])
    conjugate_input1 = torch.conj(transformed_input1)
    product = transformed_input2.mul(conjugate_input1)
    inverse_transform = torch.fft.ifftn(product, dim=[1, 2, 3])
    inverse_transform = torch.real(inverse_transform)
    return inverse_transform


def calculate_correlation(input1, input2):
    transformed_input1 = torch.fft.fft2(input1)
    transformed_input2 = torch.fft.fft2(input2)
    conjugate_input1 = torch.conj(transformed_input1)
    product = transformed_input2.mul(conjugate_input1)
    correlation = torch.fft.ifft2(product)
    correlation = torch.real(correlation)
    return correlation
