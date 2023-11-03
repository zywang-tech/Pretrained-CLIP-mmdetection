from mmdet.registry import MODELS
from mmengine.model import BaseModule
import clip
from torch.nn import BatchNorm2d
import torch

_MODELS = ['RN50x64', 'RN50x16', 'RN50x4', 'RN101', 'RN50']
@MODELS.register_module()
class Clip_RN(BaseModule):
    def __init__(self, 
                 pretrained_name,
                 num_stages=4,
                 out_indices=(0,1,2,3),
                 freeze=True,
                 frozen_stages=-1,
                 norm_eval=True,
                 ):
        super().__init__()
        assert pretrained_name != None
        assert pretrained_name in _MODELS
        self.pretrained_name = pretrained_name
        self.num_stages = num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.freeze = freeze 
        self.frozen_stages = frozen_stages
        self.num_layers = 4
        self.norm_eval = norm_eval

        self.set_backbone_model()
        self.visual = self.clip.visual
        self.text = self.clip.transformer

        # vision
        self.visual = self.clip.visual
        self.image_resolution = self.visual.input_resolution

        # text
        self.text = self.clip.transformer
        self.float()

    def set_backbone_model(self):
        self.clip, _ = clip.load(name=self.pretrained_name)
        if self.freeze:
            for _, val in self.clip.visual.named_parameters():
                val.requires_grad = False
            for _, val in self.clip.transformer.named_parameters():
                val.requires_grad = False

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in [self.visual.conv1, self.visual.conv2, self.visual.conv3,
                      self.visual.bn1,   self.visual.bn2,   self.visual.bn3, self.visual.avgpool]:
                for param in i.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self.visual, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, image):
        outs = []
        x = image
        x = self.visual.relu1(self.visual.bn1(self.visual.conv1(x)))
        x = self.visual.relu2(self.visual.bn2(self.visual.conv2(x)))
        x = self.visual.relu3(self.visual.bn3(self.visual.conv3(x)))
        x = self.visual.avgpool(x)
        
        for i, m in enumerate(range(1, self.num_layers+1)):
            res_layer = getattr(self.visual, f'layer{m}')
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        
        return tuple(outs)
    
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer freezed."""
        super(Clip_RN, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, BatchNorm2d):
                    m.eval()

    def get_per_layer_out_c(self):
        img = torch.randn(size=(1, 3, self.image_resolution, self.image_resolution)).float().cuda()
        out_c = []
        x = self.visual.relu1(self.visual.bn1(self.visual.conv1(img)))
        x = self.visual.relu2(self.visual.bn2(self.visual.conv2(x)))
        x = self.visual.relu3(self.visual.bn3(self.visual.conv3(x)))
        x = self.visual.avgpool(x)
        for i, m in enumerate(range(1, self.num_layers+1)):
            res_layer = getattr(self.visual, f'layer{m}')
            x = res_layer(x)
            out_c.append(x.shape[1])
        return out_c

    def forward_l12(self, image):
        x = image
        x = self.visual.relu1(self.visual.bn1(self.visual.conv1(x)))
        x = self.visual.relu2(self.visual.bn2(self.visual.conv2(x)))
        x = self.visual.relu3(self.visual.bn3(self.visual.conv3(x)))
        x = self.visual.avgpool(x)
        
        x = self.visual.layer1(x)
        x = self.visual.layer2(x)  
       
        return x
    
    def forward_l3(self, x):
        x = self.visual.layer3(x)
        return {"res4": x}
    
    def forward_res5(self, x):
        #detectron used last resnet layer for roi heads
        return self.visual.layer4(x)

    def attention_global_pool(self, input):
        x = input
        x = self.visual.attnpool(x)
        return x
    
    def encode_image(self, image):
        x = self.forward(image)[-1]
        x = self.attention_global_pool(x)
        return x
    
    def encode_text(self, text):
        x = self.clip.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip.text_projection
        return x

    def clip(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
    
if __name__ == '__main__':
    clip = Clip_RN(pretrained_name = 'RN50')
    clip.get_per_layer_out_c
    
