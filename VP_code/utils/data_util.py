import cv2
import random
import torch

def img2tensor(imgs, bgr2rgb=True, float32=True, gray=False):
    """Numpy array to tensor.
    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.
    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            # n = random.randint(0, 114514123)
            # cv2.imwrite(
            #     r"C:\Users\56280\Desktop\WORK-IR\Bringing-Old-Films-Back-to-Life-main\degreeoutput\example_{0}.png".format(
            #         n), img*255)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if not gray:
            img = torch.from_numpy(img.transpose(2, 0, 1))
        else:
            img = torch.from_numpy(img).unsqueeze(0)
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(imgs):
    """
    Input: t,c,h,w
    """
    def _toimg(img):

        img = torch.clamp(img, 0, 1)
        img = img.numpy().transpose(1, 2, 0)

        img = (img * 255.0).round().astype('uint8')

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    if isinstance(imgs, list):
        return [_toimg(img) for img in imgs]
    else:
        return _toimg(imgs)    


def paired_random_crop(img_gts, img_lqs, gt_patch_size_w, gt_patch_size_h, scale, gt_path):
    """Paired random crop.
    It crops lists of lq and gt images with corresponding locations.
    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.
    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size_w = gt_patch_size_w // scale
    lq_patch_size_h = gt_patch_size_h // scale

    # h_lq, w_lq, _ = img_lqs[0].shape
    # h_gt, w_gt, _ = img_gts[0].shape
    # lq_patch_size = gt_patch_size // scale

    # if h_gt != h_lq * scale or w_gt != w_lq * scale:
    #     raise ValueError(
    #         f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
    #         f'multiplication of LQ ({h_lq}, {w_lq}).')
    # if h_lq < lq_patch_size or w_lq < lq_patch_size:
    #     raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
    #                      f'({lq_patch_size}, {lq_patch_size}). '
    #                      f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size_h)
    left = random.randint(0, w_lq - lq_patch_size_w)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size_h, left:left + lq_patch_size_w, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size_h, left_gt:left_gt + gt_patch_size_w, ...]
        for v in img_gts
    ]
    # if len(img_gts) == 1:
    #     img_gts = img_gts[0]
    # if len(img_lqs) == 1:
    #     img_lqs = img_lqs[0]
    return img_gts, img_lqs


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).
    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.
    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.
    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.
    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.
    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img