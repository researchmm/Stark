from . import STARKSActor


class STARKSTActor(STARKSActor):
    """ Actor for training the STARK-ST(Stage2)"""
    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective, loss_weight, settings)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size

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
        # forward pass
        out_dict = self.forward_pass(data, run_box_head=False, run_cls_head=True)

        # process the groundtruth label
        labels = data['label'].view(-1)  # (batch, ) 0 or 1

        loss, status = self.compute_losses(out_dict, labels)

        return loss, status

    def compute_losses(self, pred_dict, labels, return_status=True):
        loss = self.loss_weight["cls"] * self.objective['cls'](pred_dict["pred_logits"].view(-1), labels)
        if return_status:
            # status for log
            status = {
                "cls_loss": loss.item()}
            return loss, status
        else:
            return loss
