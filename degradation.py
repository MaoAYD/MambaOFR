from VP_code.data.Data_Degradation.util import degradation_video_list_5
import os
import cv2
import numpy as np
import argparse


def image_read(name):
    img_lq = cv2.imread(name)
    img_lq = img_lq.astype(np.float32) / 255.
    return img_lq


def main():
    parser = argparse.ArgumentParser(description='Video degradation processing')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input directory containing sharp videos')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory for degraded videos and ground truth')

    args = parser.parse_args()

    root_path = args.input
    save_path = args.output

    video = os.listdir(root_path)
    for ii, i in enumerate(video):
        frames_names = os.listdir(os.path.join(root_path, i))
        img = []
        save_name = []
        for j in frames_names:
            img.append(image_read(os.path.join(root_path, i, j)))
            save_name.append(j)
        if ii < 10:
            degree = 0
        elif 10 <= ii < 20:
            degree = 1
        else:
            degree = 2
        degraded, gt_L = degradation_video_list_5(img, degree=degree)
        print(i, degree)

        degraded_dir = os.path.join(save_path, "degraded", i)
        gt_dir = os.path.join(save_path, "gt", i)

        if not os.path.exists(degraded_dir):
            os.makedirs(degraded_dir)
        if not os.path.exists(gt_dir):
            os.makedirs(gt_dir)

        for it, name in enumerate(save_name):
            cv2.imwrite(os.path.join(degraded_dir, name), degraded[it])
            cv2.imwrite(os.path.join(gt_dir, name), gt_L[it])


if __name__ == "__main__":
    main()