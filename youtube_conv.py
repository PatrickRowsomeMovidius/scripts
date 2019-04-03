import cv2
import os

video_fp = "/home/prowsome/Downloads/youtube.mp4"
output_dp = "/home/prowsome/Dyncal_Datasets/datasets/YouTube/large"

def create_dataset_paths(path, x_res, y_res):
    path = os.path.join(path, str(x_res)+"x"+str(y_res), "8bit")
    l_im_dir = os.path.join(path, "left")
    r_im_dir = os.path.join(path, "right")
    os.makedirs(l_im_dir)
    os.makedirs(r_im_dir)
    return l_im_dir, r_im_dir

vidcap = cv2.VideoCapture(video_fp)
success,image = vidcap.read()
count = 0
l_im_dir, r_im_dir = create_dataset_paths(  output_dp,
                                            image.shape[1]/2,
                                            image.shape[0])

while success:
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    left_im = img_gray[:, :int(img_gray.shape[1]/2)]
    right_im = img_gray[:, int(img_gray.shape[1]/2):]
    left_im_fp = os.path.join(l_im_dir, "im" + str(count) + ".png")
    right_im_fp = os.path.join(r_im_dir, "im" + str(count) + ".png")
    cv2.imwrite(left_im_fp, left_im)
    cv2.imwrite(right_im_fp, right_im)

    success, image = vidcap.read()
    count += 1
