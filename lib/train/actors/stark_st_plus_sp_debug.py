from . import BaseActor
import numpy as np
import cv2
import os
import sys


class STARKSTPLUSSPActor_debug(BaseActor):
    """ Actor for training the STARK-ST(Stage2)"""
    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        save_dir = "/data/sda/v-yanbi/iccv21/STARK_PLUS/debug_data"
        labels = data['label'].view(-1)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for idx_b in range(self.bs):
            print(idx_b, labels[idx_b].item())
            search_img = data['search_images'][0][idx_b]  # (3, 320, 320)
            search_img_np = self.de_norm(search_img.cpu().numpy().transpose((1, 2, 0)))
            x_img = cv2.cvtColor(search_img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, "%02d_x.jpg" % idx_b), x_img)
            for i in range(self.settings.num_template):
                template_img_i = data['template_images'][i][idx_b]  # (3, 128, 128)
                template_img_np = self.de_norm(template_img_i.cpu().numpy().transpose((1, 2, 0)))
                z_img = cv2.cvtColor(template_img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_dir, "%02d_%d_z.jpg" % (idx_b, i)), z_img)
        sys.exit(0)

    def de_norm(self, x):
        return (x * self.std + self.mean) * 255
