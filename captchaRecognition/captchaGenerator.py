from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import random
import cv2

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

data_path = './data/'


def random_captcha_text(char_set=number, captcha_size=4):
    """
    选择验证码上的字符
    :param char_set: 字符集
    :param captcha_size: 字符个数
    :return: 验证码上的字符列表
    """
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_capthcha_text_and_image(m):
    """
    生成验证码图片
    :param m: 验证码序号
    :return: none
    """
    image = ImageCaptcha()
    # 获取写入字符串
    captcha_text = random_captcha_text()
    captcha_text = ' '.join(captcha_text)
    captcha = image.generate(captcha_text)

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    # 写入标签
    with open(data_path + "label.txt", "a") as f:
        f.write(captcha_text)
        f.writelines("\n")
    # 写入图片
    name = data_path + 'src/' + '{0:0>4}'.format(m) + '.jpg'
    cv2.imwrite(name, captcha_image)


if __name__ == '__main__':
    for m in range(0, 5000):
        gen_capthcha_text_and_image(m)
