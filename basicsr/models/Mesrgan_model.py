import torch
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
from .srgan_model import SRGANModel


@MODEL_REGISTRY.register()
class MAESRGANModel(SRGANModel):
    """ESRGAN model for single image super-resolution."""

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        gan_gt = self.gt

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            fake_g_preds = self.net_d(self.output)
            loss_dict['l_g_gan'] = 0
            for fake_g_pred in fake_g_preds:
                l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan'] += l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_preds = self.net_d(gan_gt)
        loss_dict['l_d_real'] = 0
        loss_dict['out_d_real'] = 0
        l_d_real_tot = 0
        for real_d_pred in real_d_preds:
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            l_d_real_tot += l_d_real
            loss_dict['l_d_real'] += l_d_real
            loss_dict['out_d_real'] += torch.mean(real_d_pred.detach())
        l_d_real_tot.backward()
        # fake
        loss_dict['l_d_fake'] = 0
        loss_dict['out_d_fake'] = 0
        l_d_fake_tot = 0
        fake_d_preds = self.net_d(self.output.detach().clone())  # clone for pt1.9
        for fake_d_pred in fake_d_preds:
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            l_d_fake_tot += l_d_fake
            loss_dict['l_d_fake'] += l_d_fake
            loss_dict['out_d_fake'] += torch.mean(fake_d_pred.detach())
        l_d_fake_tot.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
