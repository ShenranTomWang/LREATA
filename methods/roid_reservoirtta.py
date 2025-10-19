import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

import math
from copy import deepcopy
from methods.base import TTAMethod
from augmentations.transforms_cotta import get_tta_transforms
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy, SymmetricCrossEntropy, SoftLikelihoodRatio
from utils.misc import ema_update_model
from methods.reservoirtta_utils import Plug_in_Bowl

@torch.no_grad()
def update_model_probs(x_ema, x, momentum=0.9):
    return momentum * x_ema + (1 - momentum) * x


@ADAPTATION_REGISTRY.register()
class ROID_ReservoirTTA(TTAMethod):
    def __init__(self, cfg, model, num_classes, scheduler: str = None):
        super().__init__(cfg, model, num_classes, scheduler=scheduler)

        self.use_weighting = cfg.ROID.USE_WEIGHTING
        self.use_prior_correction = cfg.ROID.USE_PRIOR_CORRECTION
        self.use_consistency = cfg.ROID.USE_CONSISTENCY
        self.momentum_src = cfg.ROID.MOMENTUM_SRC
        self.momentum_probs = cfg.ROID.MOMENTUM_PROBS
        self.temperature = cfg.ROID.TEMPERATURE
        self.batch_size = cfg.TEST.BATCH_SIZE
        self.e_margin = cfg.EATA.MARGIN_E0 * math.log(num_classes)   # hyper-parameter E_0 (Eqn. 3)

        
        self.class_probs_ema = [] 
        for _ in range(cfg.RESERVOIRTTA.MAX_NUM_MODELS):
            self.class_probs_ema.append(1 / self.num_classes * torch.ones(self.num_classes).to(self.device))
        
        self.tta_transform = get_tta_transforms(self.img_size, padding_mode="reflect", cotta_augs=False)

        # setup loss functions
        self.slr = SoftLikelihoodRatio()
        self.symmetric_cross_entropy = SymmetricCrossEntropy()
        self.softmax_entropy = Entropy()  # not used as loss

        # note: reduce memory consumption by only saving normalization parameters
        self.src_model = deepcopy(self.model).cpu()
        for param in self.src_model.parameters():
            param.detach_()

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.src_model, self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()


        ####################### Reservoir Start #######################

        self.reservoir = Plug_in_Bowl(cfg, self.img_size[0], self.params, 
                                 student_optimizer=self.optimizer,
                                 student_model=self.model
                                 )
        self.reservoir_output = {}
        ####################### Reservoir End #######################

    def loss_calculation(self, x):
        imgs_test = x[0]

        ####################### Reservoir Start #######################
        self.reservoir_output = self.reservoir.clustering(imgs_test)
        
        with torch.no_grad():
            self.reservoir(ensembling=True, which_model='student')
            ensembled_outputs = self.model(imgs_test)

        self.reservoir(ensembling=False, which_model='student')

        self.optimizer.load_state_dict(self.reservoir.student.optimizer_reservoir[self.reservoir.model_idx])
        self.optimizer.zero_grad()
        ####################### Reservoir End #######################
        
        outputs = self.model(imgs_test)

        perform_update = True
        if self.use_weighting:
            with torch.no_grad():
                # calculate diversity based weight
                weights_div = 1 - F.cosine_similarity(self.class_probs_ema[self.reservoir.model_idx].unsqueeze(dim=0), outputs.softmax(1), dim=1)
                weights_div = (weights_div - weights_div.min()) / (weights_div.max() - weights_div.min())
                mask = weights_div < weights_div.mean()

                # calculate certainty based weight
                weights_cert = - self.softmax_entropy(logits=outputs)
                weights_cert = (weights_cert - weights_cert.min()) / (weights_cert.max() - weights_cert.min())
                if self.cfg.MODEL.ARCH == "Standard_VITB":
                    mask &= (-weights_cert >= self.e_margin)

                # calculate the final weights
                weights = torch.exp(weights_div * weights_cert / self.temperature)
                weights[mask] = 0.
                perform_update = sum(weights) > 0


                self.class_probs_ema[self.reservoir.model_idx] = update_model_probs(x_ema=self.class_probs_ema[self.reservoir.model_idx], x=outputs.softmax(1).mean(0), momentum=self.momentum_probs)

        # calculate the soft likelihood ratio loss
        if perform_update:
            if self.cfg.MODEL.ARCH == "Standard_VITB":
                loss_out = self.softmax_entropy(logits=outputs)
            else:
                loss_out = self.slr(logits=outputs)

            # weight the loss
            if self.use_weighting:
                loss_out = loss_out * weights
                loss_out = loss_out[~mask]
            loss = loss_out.sum() / self.batch_size

            # calculate the consistency loss
            if self.use_consistency:
                outputs_aug = self.model(self.tta_transform(imgs_test[~mask]))
                loss += (self.symmetric_cross_entropy(x=outputs_aug, x_ema=outputs[~mask]) * weights[~mask]).sum() / self.batch_size

        return ensembled_outputs, loss if perform_update else torch.Tensor([0.]), perform_update

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss, perform_update = self.loss_calculation(x)
            if perform_update:
                loss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()

        if perform_update:
            self.model = ema_update_model(
                model_to_update=self.model,
                model_to_merge=self.src_model,
                momentum=self.momentum_src,
                device=self.device
            )

        with torch.no_grad():
            if self.use_prior_correction:
                prior = outputs.softmax(1).mean(0)
                smooth = max(1 / outputs.shape[0], 1 / outputs.shape[1]) / torch.max(prior)
                smoothed_prior = (prior + smooth) / (1 + smooth * outputs.shape[1])
                outputs *= smoothed_prior

        ####################### Reservoir Start #######################
        self.reservoir.update_kth_model(self.optimizer, which_model='student', which_part='params')
        ####################### Reservoir End #######################
        
        return outputs

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()

        for _ in range(self.cfg.RESERVOIRTTA.MAX_NUM_MODELS):
            self.class_probs_ema.append(1 / self.num_classes * torch.ones(self.num_classes).to(self.device))

    def collect_params(self):
        """Collect the affine scale + shift parameters from normalization layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model."""
        self.model.eval()   # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        self.model.requires_grad_(False)  # disable grad, to (re-)enable only necessary parts
        # re-enable gradient for normalization layers
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
