import torch
import torch.nn.functional as F
from torch import Tensor, einsum, nn
import random
import numpy as np
from math import pi
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union
from torchmetrics import MeanMetric
import pytorch_lightning as pl
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Cyclical_embedding(nn.Module):
    def __init__(self, frequencies: list):
        super().__init__()
        self.frequencies = frequencies
        self.dim = len(self.frequencies) * 2

    def forward(self, time_coords: torch.Tensor):
        """
        Args:
            time_coords (torch.Tensor): Time coordinates of shape [B, T, C, H, W]
        """
        embeddings = []
        for i, frequency in enumerate(self.frequencies):
            embeddings += [
                torch.sin(2 * torch.pi * time_coords[:, :, i] / frequency),
                torch.cos(2 * torch.pi * time_coords[:, :, i] / frequency),
            ]
        embeddings = torch.stack(embeddings, axis=2)
        return embeddings


class RoCrossViViT(nn.Module):
    def __init__(
        self,
        image_size = [8, 8],
        patch_size = [1, 1],
        time_coords_encoder = Cyclical_embedding([12, 31, 24, 60]),
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: int = 4,
        ctx_channels: int = 27,
        ts_channels: int = 1,
        ts_length: int = 48,
        out_dim: int = 1,
        dim_head: int = 64,
        dropout: float = 0.0,
        freq_type: str = "lucidrains",
        pe_type: str = "rope",
        num_mlp_heads: int = 2,
        use_glu: bool = True,
        ctx_masking_ratio: float = 0.9,
        ts_masking_ratio: float = 0.9,
        decoder_dim: int = 128,
        decoder_depth: int = 4,
        decoder_heads: int = 6,
        decoder_dim_head: int = 128,
        **kwargs,
    ):
        super().__init__()
        assert (
            ctx_masking_ratio >= 0 and ctx_masking_ratio < 1
        ), "ctx_masking_ratio must be in [0,1)"
        assert pe_type in [
            "rope",
            "sine",
            "learned",
            None,
        ], f"pe_type must be 'rope', 'sine', 'learned' or None but you provided {pe_type}"
        self.time_coords_encoder = time_coords_encoder
        self.ctx_channels = ctx_channels
        self.ts_channels = ts_channels
        if hasattr(self.time_coords_encoder, "dim"):
            self.ctx_channels += self.time_coords_encoder.dim
            self.ts_channels += self.time_coords_encoder.dim

        self.image_size = image_size
        self.patch_size = patch_size
        self.ctx_masking_ratio = ctx_masking_ratio
        self.ts_masking_ratio = ts_masking_ratio
        self.num_mlp_heads = num_mlp_heads
        self.pe_type = pe_type

        for i in range(2):
            ims = self.image_size[i]
            ps = self.patch_size[i]
            assert (
                ims % ps == 0
            ), "Image dimensions must be divisible by the patch size."

        patch_dim = self.ctx_channels * self.patch_size[0] * self.patch_size[1]
        num_patches = (self.image_size[0] // self.patch_size[0]) * (
            self.image_size[1] // self.patch_size[1]
        )

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
            ),
            nn.Linear(patch_dim, dim),
        )
        kwargs["max_freq"] = 128
        self.enc_pos_emb = AxialRotaryEmbedding(dim_head, freq_type, **kwargs)
        self.ts_embedding = nn.Linear(self.ts_channels, dim)
        self.ts_downsampler = nn.Conv1d(1, 1, 2, 2)
        self.ctx_encoder = VisionTransformer(
            dim,
            depth,
            heads,
            dim_head,
            dim * mlp_ratio,
            image_size,
            dropout,
            pe_type == "rope",
            use_glu,
        )
        if pe_type == "learned":
            self.pe_ctx = nn.Parameter(torch.randn(1, num_patches, dim))
            self.pe_ts = nn.Parameter(torch.randn(1, 1, dim))
        elif pe_type == "sine":
            self.pe_ctx = PositionalEncoding2D(dim)
            self.pe_ts = PositionalEncoding2D(dim)
        self.mixer = CrossTransformer(
            dim,
            depth,
            heads,
            dim_head,
            dim * mlp_ratio,
            image_size,
            dropout,
            pe_type == "rope",
            use_glu,
        )
        self.ctx_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.ts_encoder = Transformer(
            dim,
            ts_length,
            depth,
            heads,
            dim_head,
            dim * mlp_ratio,
            dropout=dropout,
        )
        self.ts_enctodec = nn.Linear(dim, decoder_dim)
        self.temporal_transformer = Transformer(
            decoder_dim,
            ts_length,
            decoder_depth,
            decoder_heads,
            decoder_dim_head,
            decoder_dim * mlp_ratio,
            dropout=dropout,
        )
        self.ts_mask_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.mlp_heads = nn.ModuleList([])
        for i in range(num_mlp_heads):
            self.mlp_heads.append(
                nn.Sequential(
                    nn.LayerNorm(decoder_dim),
                    nn.Linear(decoder_dim, out_dim, bias=True),
                    nn.ReLU(),
                )
            )

        self.quantile_masker = nn.Sequential(
            nn.Conv1d(decoder_dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            Rearrange(
                "b c t -> b t c",
            ),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_mlp_heads),
        )

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward(
        self,
        ctx: torch.Tensor,
        ctx_coords: torch.Tensor,
        ts: torch.Tensor,
        ts_coords: torch.Tensor,
        time_coords_ctx: torch.Tensor,
        time_coords_ts: torch.Tensor,
        mask: bool = True,
    ):
        """
        Args:
            ctx (torch.Tensor): Context frames of shape [B, T, C, H, W]
            ctx_coords (torch.Tensor): Coordinates of context frames of shape [B, 2, H, W]
            ts (torch.Tensor): Station timeseries of shape [B, T, C]
            ts_coords (torch.Tensor): Station coordinates of shape [B, 2, 1, 1]
            time_coords_ctx (torch.Tensor): Time coordinates of shape [B, T, C, H, W] in every 1 hour
            time_coords_ts (torch.Tensor): Time coordinates of shape [B, T, C, H, W] in every 30 mins
            Note that time_coords_ctx/ts must have the same length for otherwise CA's spatial attention won't work.
            mask (bool): Whether to mask or not. Useful for inference
        Returns:

        """
        B, T, _, H, W = ctx.shape
        ts = self.ts_downsampler(ts.transpose(-1, -2)).transpose(-1, -2)
        time_coords_ctx = self.time_coords_encoder(time_coords_ctx)
        time_coords_ts = self.time_coords_encoder(time_coords_ts)

        ctx = torch.cat([ctx, time_coords_ctx], axis=2)
        ts = torch.cat([ts, time_coords_ts[..., 0, 0]], axis=-1)
        ctx = rearrange(ctx, "b t c h w -> (b t) c h w")

        ctx_coords = repeat(ctx_coords, "b c h w -> (b t) c h w", t=T)
        ts_coords = repeat(ts_coords, "b c h w -> (b t) c h w", t=T)
        src_enc_pos_emb = self.enc_pos_emb(ctx_coords)
        tgt_pos_emb = self.enc_pos_emb(ts_coords)

        ctx = self.to_patch_embedding(ctx)  # BT, N, D
        if self.pe_type == "learned":
            ctx = ctx + self.pe_ctx
        elif self.pe_type == "sine":
            pe = self.pe_ctx(ctx_coords)
            pe = rearrange(pe, "b h w c -> b (h w) c")
            ctx = ctx + pe
        if self.ctx_masking_ratio > 0 and mask:
            p = self.ctx_masking_ratio * random.random()
            ctx, _, ids_restore, ids_keep = self.random_masking(ctx, p)
            src_enc_pos_emb = tuple(
                torch.gather(
                    pos_emb,
                    dim=1,
                    index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_emb.shape[-1]),
                )
                for pos_emb in src_enc_pos_emb
            )
        latent_ctx, self_attention_scores = self.ctx_encoder(ctx, src_enc_pos_emb)

        ts = self.ts_embedding(ts)
        if self.ts_masking_ratio > 0 and mask:
            p = self.ts_masking_ratio * random.random()
            ts, _, ids_restore, ids_keep = self.random_masking(ts, p)
            mask_tokens = self.ts_mask_token.repeat(ts.shape[0], T - ts.shape[1], 1)
            ts = torch.cat([ts, mask_tokens], dim=1)
            ts = torch.gather(
                ts, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, ts.shape[2])
            )

        latent_ts = self.ts_encoder(ts)
        latent_ts = rearrange(latent_ts, "b t c -> (b t) c").unsqueeze(1)

        if self.pe_type == "learned":
            latent_ts = latent_ts + self.pe_ts
        elif self.pe_type == "sine":
            pe = self.pe_ts(ts_coords)
            pe = rearrange(pe, "b h w c -> b (h w) c")
            latent_ts = latent_ts + pe
        latent_ts, cross_attention_scores = self.mixer(
            latent_ctx, latent_ts, src_enc_pos_emb, tgt_pos_emb
        )
        latent_ts = latent_ts.squeeze(1)
        latent_ts = self.ts_enctodec(rearrange(latent_ts, "(b t) c -> b t c", b=B))

        y = self.temporal_transformer(latent_ts)

        # Handles the multiple MLP heads
        outputs = []
        for i in range(self.num_mlp_heads):
            mlp = self.mlp_heads[i]
            output = mlp(y)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=2)
        outputs = outputs.reshape(B, -1)  # [B, 2T]

        quantile_mask = self.quantile_masker(rearrange(y.detach(), "b t c -> b c t"))

        return (outputs, quantile_mask, self_attention_scores, cross_attention_scores)


class ContextMixerModule(ABC, pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        metrics: dict,
        criterion: torch.nn.Module,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["criterion"])

        self.model = model
        self.criterion = criterion

        self._set_metrics()

    def _set_metrics(self):
        if self.hparams.metrics.train is not None:
            for k in self.hparams.metrics.train:
                setattr(self, f"train_{k}", self.hparams.metrics.train[k])
        if self.hparams.metrics.val is not None:
            for k in self.hparams.metrics.val:
                setattr(self, f"val_{k}", self.hparams.metrics.val[k])

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def calc_dot_product(
        self,
        optflow_data: torch.Tensor,
        coords: torch.Tensor,
        station_coords: torch.Tensor,
    ):
        """
        Args:
            optflow_data: Optical flow data. Tensor of shape [B, T, 2*C, H, W]
            coords: Coordinates of each pixel. Tensor of shape [B, 2, H, W]
            station_coords: Coordinates of the station. Tensor of shape [B, 2, 1, 1]
        Returns:
            dp: Dot product between the optical flow and station vectors of shape [B, T, C, H, W]
        """
        optflow_data = rearrange(optflow_data, "b t (c n) h w -> b t c n h w", n=2)
        dist = station_coords - coords
        dist = repeat(
            dist,
            "b n h w -> b t c n h w",
            t=optflow_data.shape[1],
            c=optflow_data.shape[2],
        )
        optflow_data = F.normalize(
            rearrange(optflow_data, "b t c n h w -> b t c h w n"), p=2, dim=-1
        )
        dist = F.normalize(rearrange(dist, "b t c n h w -> b t c h w n"), p=2, dim=-1)
        dp = optflow_data[..., 0] * dist[..., 0] + optflow_data[..., 1] * dist[..., 1]
        return dp

    def prepare_batch(self, batch, use_target=True):
        x_ctx = batch["context"].float()
        x_opt = batch["optical_flow"].float()
        x_ts = batch["timeseries"].float()
        if use_target:
            y_ts = batch["target"].float()
            y_previous_ts = batch["target_previous"].float()
        spatial_coords = batch["spatial_coordinates"].float()
        time_coords = batch["time_coordinates"].float()
        ts_coords = batch["station_coords"].float()
        ts_elevation = batch["station_elevation"].float()

        aux_data = []
        for k in batch["auxiliary_data"]:
            ad = batch["auxiliary_data"][k].float()  # B, H, W
            ad = ad.unsqueeze(1).unsqueeze(1)
            ad = ad.repeat(1, x_ctx.shape[1], 1, 1, 1)
            aux_data.append(ad)
        x_ctx = torch.cat(
            [
                x_ctx,
            ]
            + aux_data,
            axis=2,
        )

        if self.hparams.use_dp:
            x_dp = self.calc_dot_product(x_opt, spatial_coords, ts_coords)
            x_ctx = torch.cat([x_ctx, x_dp], axis=2)
        else:
            x_ctx = torch.cat([x_ctx, x_opt], axis=2)
        ts_elevation = ts_elevation[..., 0, 0]
        ts_elevation = ts_elevation[(...,) + (None,) * 2]
        ts_elevation = ts_elevation.repeat(1, x_ts.shape[1], 1)
        x_ts = torch.cat([x_ts, ts_elevation], axis=-1)

        H, W = x_ctx.shape[-2:]
        ctx_coords = F.interpolate(
            spatial_coords,
            size=(H // self.model.patch_size[0], W // self.model.patch_size[1]),
            mode="bilinear",
        )
        if use_target:
            return x_ts, x_ctx, y_ts, y_previous_ts, ctx_coords, ts_coords, time_coords
        return x_ts, x_ctx, ctx_coords, ts_coords, time_coords

    @abstractmethod
    def forward(
        self,
        x_ctx: Tensor,
        ctx_coords: Tensor,
        x_ts: Tensor,
        ts_coords: Tensor,
        time_coords: Tensor,
    ) -> Tensor:
        pass

    @abstractmethod
    def training_step(self, train_batch, batch_idx) -> Any:
        pass

    @abstractmethod
    def validation_step(self, val_batch, batch_idx) -> Any:
        pass

class CrossViViT(ContextMixerModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        metrics: dict,
        criterion: torch.nn.Module,
        **kwargs,
    ):
        super().__init__(model, optimizer, scheduler, metrics, criterion, kwargs=kwargs)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask):
        out, _, self_attention_scores, cross_attention_scores = self.model(
            x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask
        )
        return out

    def training_step(self, train_batch, batch_idx):
        (
            x_ts,
            x_ctx,
            y_ts,
            y_prev_ts,
            ctx_coords,
            ts_coords,
            time_coords,
        ) = self.prepare_batch(train_batch)

        y_hat = self(x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=True)
        y_hat = y_hat.mean(dim=2)

        loss = self.criterion(y_hat, y_ts)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, prog_bar=True)

        for key in self.hparams.metrics.train:
            metric = getattr(self, f"train_{key}")
            if hasattr(metric, "needs_previous") and metric.needs_previous:
                metric(y_hat, y_ts, y_prev_ts)
            else:
                metric(y_hat, y_ts)
            self.log(
                f"train/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def validation_step(self, val_batch, batch_idx):
        (
            x_ts,
            x_ctx,
            y_ts,
            y_prev_ts,
            ctx_coords,
            ts_coords,
            time_coords,
        ) = self.prepare_batch(val_batch)

        y_hat = self(x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=False)
        y_hat = y_hat.mean(dim=2)

        loss = self.criterion(y_hat, y_ts)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, prog_bar=True)

        for key in self.hparams.metrics.val:
            metric = getattr(self, f"val_{key}")
            if hasattr(metric, "needs_previous") and metric.needs_previous:
                metric(y_hat, y_ts, y_prev_ts)
            else:
                metric(y_hat, y_ts)
            self.log(
                f"val/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return {"predictions": y_hat, "ground_truth": y_ts}


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


def rotate_every_two(x):
    x = rearrange(x, "... (d j) -> ... d j", j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d j -> ... (d j)")


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, coords):
        """
        :param tensor: A 4d tensor of size (batch_size, ch, x, y)
        :param coords: A 4d tensor of size (batch_size, num_coords, x, y)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(coords.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        batch_size, _, x, y = coords.shape
        self.cached_penc = None
        pos_x = coords[:, 0, 0, :].type(self.inv_freq.type())  # batch, width
        pos_y = coords[:, 1, :, 0].type(self.inv_freq.type())  # batch, height
        sin_inp_x = torch.einsum("bi,j->bij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("bi,j->bij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(2)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb = torch.zeros(
            (batch_size, x, y, self.channels * 2), device=coords.device
        ).type(coords.type())
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y

        return emb


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class CrossPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_src = nn.LayerNorm(dim)
        self.norm_tgt = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, ctx, src_pos_emb, ts, tgt_pos_emb):
        return self.fn(self.norm_src(ctx), src_pos_emb, self.norm_tgt(ts), tgt_pos_emb)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, use_glu=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2 if use_glu else hidden_dim),
            GEGLU() if use_glu else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        use_rotary=True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.use_rotary = use_rotary
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, pos_emb):
        """
        Args:
            x: Sequence of shape [B, N, D]
            pos_emb: Positional embedding of sequence's tokens of shape [B, N, D]
        """

        q = self.to_q(x)

        qkv = (q, *self.to_kv(x).chunk(2, dim=-1))
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.heads), qkv
        )

        if self.use_rotary:

            sin, cos = map(
                lambda t: repeat(t, "b n d -> (b h) n d", h=self.heads), pos_emb
            )
            dim_rotary = sin.shape[-1]

            # handle the case where rotary dimension < head dimension

            (q, q_pass), (k, k_pass) = map(
                lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k)
            )
            q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
            q, k = map(lambda t: torch.cat(t, dim=-1), ((q, q_pass), (k, k_pass)))

        dots = einsum("b i d, b j d -> b i j", q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)
        return self.to_out(out), attn


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        use_rotary=True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.use_rotary = use_rotary
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, src, src_pos_emb, tgt, tgt_pos_emb):

        q = self.to_q(tgt)

        qkv = (q, *self.to_kv(src).chunk(2, dim=-1))

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.heads), qkv
        )

        if self.use_rotary:
            # apply 2d rotary embeddings to queries and keys

            sin_src, cos_src = map(
                lambda t: repeat(t, "b n d -> (b h) n d", h=self.heads), src_pos_emb
            )
            sin_tgt, cos_tgt = map(
                lambda t: repeat(t, "b n d -> (b h) n d", h=self.heads), tgt_pos_emb
            )
            dim_rotary = sin_src.shape[-1]

            # handle the case where rotary dimension < head dimension

            (q, q_pass), (k, k_pass) = map(
                lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k)
            )
            q = (q * cos_tgt) + (rotate_every_two(q) * sin_tgt)
            k = (k * cos_src) + (rotate_every_two(k) * sin_src)
            q, k = map(lambda t: torch.cat(t, dim=-1), ((q, q_pass), (k, k_pass)))

        dots = einsum("b i d, b j d -> b i j", q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)
        return self.to_out(out), attn


class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, freq_type="lucidrains", **kwargs):
        super().__init__()
        self.dim = dim
        self.freq_type = freq_type
        if freq_type == "lucidrains":
            scales = torch.linspace(1.0, kwargs["max_freq"] / 2, self.dim // 4)
        elif freq_type == "vaswani":
            scales = 1 / (
                kwargs["base"] ** (torch.arange(0, self.dim, 4).float() / self.dim)
            )
        else:
            NotImplementedError(
                f"Only 'lucidrains' and 'vaswani' frequencies are implemented, but you chose {freq_type}."
            )
        self.register_buffer("scales", scales)

    def forward(self, coords: torch.Tensor):
        """
        Assumes that coordinates do not change throughout the batches.
        Args:
            coords (torch.Tensor): Coordinates of shape [B, 2, H, W]
        """
        seq_x = coords[:, 0, 0, :]
        seq_x = seq_x.unsqueeze(-1)
        seq_y = coords[:, 1, :, 0]
        seq_y = seq_y.unsqueeze(-1)

        scales = self.scales[(*((None, None)), Ellipsis)]
        scales = scales.to(coords)

        if self.freq_type == "lucidrains":
            seq_x = seq_x * scales * pi
            seq_y = seq_y * scales * pi
        elif self.freq_type == "vaswani":
            seq_x = seq_x * scales
            seq_y = seq_y * scales

        x_sinu = repeat(seq_x, "b i d -> b i j d", j=seq_y.shape[1])
        y_sinu = repeat(seq_y, "b j d -> b i j d", i=seq_x.shape[1])

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim=-1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim=-1)

        sin, cos = map(lambda t: rearrange(t, "b i j d -> b (i j) d"), (sin, cos))
        sin, cos = map(lambda t: repeat(t, "b n d -> b n (d j)", j=2), (sin, cos))
        return sin, cos


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, num_frames, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, T, C]
        """
        x += self.pos_embedding
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        image_size: Union[List[int], Tuple[int], int],
        dropout: float = 0.0,
        use_rotary: bool = True,
        use_glu: bool = True,
    ):
        super().__init__()
        self.image_size = image_size

        self.blocks = nn.ModuleList([])

        for _ in range(depth):
            self.blocks.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            SelfAttention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                use_rotary=use_rotary,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout=dropout, use_glu=use_glu),
                        ),
                    ]
                )
            )

    def forward(
        self,
        src: torch.Tensor,
        src_pos_emb: torch.Tensor,
    ):
        """
        Performs the following computation in each layer:
            1. Self-Attention on the source sequence
            2. FFN on the source sequence
        Args:
            src: Source sequence of shape [B, N, D]
            src_pos_emb: Positional embedding of source sequence's tokens of shape [B, N, D]
        """

        attention_scores = {}
        for i in range(len(self.blocks)):
            sattn, sff = self.blocks[i]

            out, sattn_scores = sattn(src, pos_emb=src_pos_emb)
            attention_scores["self_attention"] = sattn_scores
            src = out + src
            src = sff(src) + src

        return src, attention_scores


class CrossTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        image_size: Union[List[int], Tuple[int], int],
        dropout: float = 0.0,
        use_rotary: bool = True,
        use_glu: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.cross_layers = nn.ModuleList([])

        for _ in range(depth):
            self.cross_layers.append(
                nn.ModuleList(
                    [
                        CrossPreNorm(
                            dim,
                            CrossAttention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                use_rotary=use_rotary,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout=dropout, use_glu=use_glu),
                        ),
                    ]
                )
            )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_pos_emb: torch.Tensor,
        tgt_pos_emb: torch.Tensor,
    ):
        """
        Performs the following computation in each layer:
            1. Self-Attention on the source sequence
            2. FFN on the source sequence
            3. Cross-Attention between target and source sequence
            4. FFN on the target sequence
        Args:
            src: Source sequence of shape [B, N, D]
            tgt: Target sequence of shape [B, M, D]
            src_pos_emb: Positional embedding of source sequence's tokens of shape [B, N, D]
            tgt_pos_emb: Positional embedding of target sequence's tokens of shape [B, M, D]
        """

        attention_scores = {}
        for i in range(len(self.cross_layers)):
            cattn, cff = self.cross_layers[i]
            out, cattn_scores = cattn(src, src_pos_emb, tgt, tgt_pos_emb)
            attention_scores["cross_attention"] = cattn_scores
            tgt = out + tgt
            tgt = cff(tgt) + tgt

        return tgt, attention_scores
