__author__ = 'kolosnjaji'

import PIL
from PIL import Image
from subprocess import Popen, PIPE
import numpy as np

def resize_images(size=(300,300)):

    image_dirs = open('/home/kolosnjaji/datasets/crowdsourcing/Welinder/attributes/attributes/images-dirs.txt').readlines()

    for image_str in image_dirs:
        image_vars = image_str.split()
        image_id = image_vars[0]
        image_path = '/home/kolosnjaji/datasets/crowdsourcing/Welinder/images/' + image_vars[1]
        my_img = Image.open(image_path)

        my_img = my_img.resize(size, PIL.Image.ANTIALIAS)
        my_img.save('/home/kolosnjaji/papers/communication_efficient_ensemble_learning/github/LowBudgetGP/data/welinder/images/' + str(image_id) + '.jpg')


if __name__ == "__main__":
    #resize_images()
    num_images = 6033

    for i in range(num_images):
        print i
        cmd = "/home/kolosnjaji/papers/communication_efficient_ensemble_learning/OverFeat/bin/linux_64/overfeat -f /home/kolosnjaji/papers/communication_efficient_ensemble_learning/github/LowBudgetGP/data/welinder/images/{0}.jpg".format(i)
        print cmd
        p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
        stdout, stderr = p.communicate()
        stdout_lines = stdout.split('\n')
        stdout_feat_list = stdout_lines[1].split(' ')
        stdout_feat_list_float = [float(stdout_feat_list[k]) for k in range(len(stdout_feat_list[0:-1]))]
        np.savetxt('/home/kolosnjaji/papers/communication_efficient_ensemble_learning/github/LowBudgetGP/data/welinder/images/{0}'.format(i) + '.txt', np.asarray(stdout_feat_list_float))

#        ('/home/kolosnjaji/papers/communication_efficient_ensemble_learning/github/LowBudgetGP/data/welinder/images/{0}'.format(i) + '.txt'))

