from mmdet.registry import MODELS
from mmengine.model import BaseModule
import clip
import torch
# from mmdet.models.detectors.base_detr import DetectionTransformer

_MODELS = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

@MODELS.register_module()
class Clip_ViT(BaseModule):
    def __init__(self, 
                 pretrained_name,
                 freeze=True,
                 frozen_stages=-1,
                 return_intermediate=False,
                 ):
        super().__init__()
        assert pretrained_name != None
        assert pretrained_name in _MODELS
        self.pretrained_name = pretrained_name
        self.freeze = freeze 
        self.frozen_stages = frozen_stages
        self.num_layers = 4
        self.return_intermediate = return_intermediate

        self.set_backbone_model()
        self.visual = self.clip.visual
        self.text = self.clip.transformer
        self.logit_scale = self.clip.logit_scale

        # vision
        self.visual = self.clip.visual
        self.image_resolution = self.visual.input_resolution
        self.output_dim = self.visual.output_dim

        # text
        self.text = self.clip.transformer
        self.dtype = float
        self.float()

    def set_backbone_model(self):
        self.clip, _ = clip.load(name=self.pretrained_name)
        if self.freeze:
            for _, val in self.clip.visual.named_parameters():
                val.requires_grad = False
            for _, val in self.clip.transformer.named_parameters():
                val.requires_grad = False

    def forward(self, img):
        intermediate = []
        x = self.visual.conv1(img)  
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)  
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 
                         1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(0, 4):
            layer = self.visual.transformer.resblocks[i]
            x = layer(x)
            NLD = x.permute(1, 0, 2)  # LND -> NLD
            cls_token = self.visual.ln_post(NLD[:, 0, :]) #取class token
            if self.return_intermediate:
                intermediate.append(cls_token)
    
        if self.return_intermediate:
            return torch.stack(intermediate)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.visual.ln_post(x[:, 0, :]) #取class token

        if self.visual.proj is not None:
            x = x @ self.visual.proj
        return x
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in [self.visual.conv1, 
                      self.visual.class_embedding, 
                      self.visual.positional_embedding,
                      self.visual.proj]:
                for param in i.parameters():
                        param.requires_grad = False

        for i in range(0, self.frozen_stages):
            m = self.visual.transformer.resblocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(Clip_ViT, self).train(mode)
        self._freeze_stages()

    def get_per_layer_out_c(self):
        img = torch.randn(size=(1, 3, self.image_resolution, self.image_resolution)).float().cuda()
        out_c = []
        x = self.visual.conv1(img)  
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)  
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 
                         1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(0, self.num_layers):
            transformer = self.visual.transformer.resblocks[i]
            x = transformer(x)
            out_c.append(x.shape[-1])
        return out_c
    
    def encode_image(self, image):
        if self.return_intermediate:
            img = self.forward(image)[-1]
            if self.visual.proj is not None:
                img = img @ self.visual.proj
            return img
        else:
            return self.forward(image)
    
    def tokenize(self, text: list):
        return clip.tokenize(text)
    
    def encode_text(self, text):
        x = self.clip.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.clip.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip.text_projection
        return x

    def clip_img_text(self, image, text):
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
    clip1 = Clip_ViT(pretrained_name = 'ViT-B/16', return_intermediate=True)
    img = torch.randn(size=(1, 3, 224, 224)).cuda().float()
    text = ['hello']
    text_ = clip1.tokenize(text)
    print(clip1.clip_img_text(img, text_.cuda()))
    
