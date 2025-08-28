import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 /
                                             self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        return encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.register_buffer('_embedding', torch.empty(
            self._num_embeddings, self._embedding_dim))
        self._embedding.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', torch.empty(
            num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward_indice(self, encoding_indices):
        input_shape = encoding_indices.shape
        encoding_indices = encoding_indices.reshape((-1, 1))
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=encoding_indices.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding).view(
            input_shape[0], input_shape[1], input_shape[2], self._embedding_dim)

        return quantized.permute(0, 3, 1, 2).contiguous()

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # convert quantized from BHWC -> BCHW
        return encoding_indices.view(input_shape[0:3])


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


class ResBlock_v1(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class ResBlock_v2(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1, bias=False),
            nn.BatchNorm2d(chan),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1, bias=False),
            nn.BatchNorm2d(chan),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + x)


class VQVAE(nn.Module):
    def __init__(
        self,
        image_size=256,
        num_tokens=512,
        codebook_dim=512,
        num_resnet_blocks=2,
        hidden_dim=16,
        channels=1,
        loss_type='mse',
        temperature=0.9,
        straight_through=False,
        kl_div_loss_weight=0.,
        simplify=False,
        max_value=1.0,
        downsample_ratio=16,
        use_norm=True,
        alpha=0.25,
        train_objective="regression",
        num_classes=2,
        residul_type='v1',
        ema_decay=0.99,
        commitment_cost=0.25,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        # hidden_dim = hidden_dim // (downsample_ratio // 2)
        self.num_resnet_blocks = num_resnet_blocks
        self.simplify = simplify
        self.image_size = image_size
        self.num_tokens = num_tokens
        self.temperature = temperature
        self.straight_through = straight_through
        self.downsample_ratio = downsample_ratio
        self.layer_num = int((torch.log2(torch.tensor(downsample_ratio)) - 1).item())
        self.use_norm = use_norm
        self.alpha = alpha,
        self.loss_type = loss_type
        self.train_objective = train_objective
        self.num_classes = num_classes
        self.max_value = max_value
        self.residul_type = residul_type
        if self.residul_type == 'v1':
            ResBlock = ResBlock_v1
        elif self.residul_type == 'v2':
            ResBlock = ResBlock_v2

        dim = self.hidden_dim
        enc_layers = [
            nn.Conv2d(channels, dim, 4, stride=2, padding=1), nn.ReLU()]

        for i in range(self.layer_num):
            enc_layers.append(nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1))
            enc_layers.append(nn.ReLU())
            dim = dim * 2

        for i in range(num_resnet_blocks):
            enc_layers.append(ResBlock(dim))
        enc_layers.append(nn.Conv2d(dim, codebook_dim, 1))

        dim = self.hidden_dim * self.downsample_ratio // 2

        dec_layers = [nn.Conv2d(codebook_dim, dim, 1), nn.ReLU()]

        for i in range(num_resnet_blocks):
            dec_layers.append(ResBlock(dim))

        dec_layers.append(nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1))
        dec_layers.append(nn.ReLU())
        for i in range(self.layer_num):
            dec_layers.append(nn.ConvTranspose2d(
                dim, dim // 2, 4, stride=2, padding=1))
            dec_layers.append(nn.ReLU())
            dim = dim // 2
        dec_layers.append(nn.Conv2d(dim, channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)


        if ema_decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_tokens, codebook_dim,
                                              commitment_cost, ema_decay)
        else:
            self._vq_vae = VectorQuantizer(num_tokens, codebook_dim,
                                           commitment_cost)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            pretrained_model = torch.load(pretrained)
            # remove module. prefix
            pretrained_model['weights'] = {k.replace('module.', ''): v for k, v in pretrained_model['weights'].items()}
            self.load_state_dict(pretrained_model['weights'])

    def norm(self, images):
        images = 2. * (images / self.max_value - 0.5)
        return images

    def denorm(self, images):
        images = images * 0.5 + 0.5
        return images * self.max_value

    @torch.no_grad()
    def get_codebook_indices(self, images):
        return self(img=images, return_indices=True)
    
    @torch.no_grad()
    def encode(self, images):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            indices = self(img=images, return_indices=True)
        return indices

    @torch.no_grad()
    def decode(self, indices):
        with torch.inference_mode():
            quantized = self._vq_vae.forward_indice(indices)
            images = self.decoder(quantized)
        return images


    def forward(
        self,
        img,
        label_img=None,
        use_norm=None,
        return_indices=False
    ):
        img = img.float()

        use_norm = self.use_norm if use_norm is None else use_norm
        if use_norm:
            img = self.norm(img)

        logits = self.encoder(img)
        code_indices = self._vq_vae(logits)

        if return_indices:
            return code_indices


def get_vqvae_processor():
    model = dict(
        image_size=128,
        num_resnet_blocks=2,
        downsample_ratio=16,
        num_tokens=128,
        codebook_dim=512,
        hidden_dim=16,
        channels=1,
        use_norm=True,
        train_objective='classification',
        max_value=1.,
        residul_type='v1',
        loss_type='bce',
    )
    vae = VQVAE(**model)
    ckpt_path = "/jfs/jensen/code/LLaMA-Factory/data/vae_0_40000.pt"
    vae.init_weights(ckpt_path)
    vae.eval()

    # vae.to(torch.cuda.current_device())
    return vae