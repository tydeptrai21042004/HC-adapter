# models/hcc_adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import gcd

class HCCAdapter(nn.Module):
    def __init__(
        self, C, M=1, h=1, axis='hw',
        alpha_group=16,             # NEW API (group-shared alphas)
        tie_sym=True,
        no_pw=False,                # NEW API (inverse of legacy use_pw)
        pw_ratio=32,
        pw_groups=4,
        use_bn=False,
        residual_scale=1.0,
        gate_init=0.1,
        padding_mode='reflect',
        **legacy,                   # <-- accept unknown legacy kwargs
    ):
        super().__init__()
        assert axis in ('h','w','hw')
        self.C, self.M, self.h = int(C), int(M), int(h)
        self.axis = axis
        self.tie_sym = tie_sym
        self.padding_mode = padding_mode
        self.residual_scale = residual_scale

        # --- translate legacy args ---
        # per_channel=True  -> alpha_group = 1
        if 'per_channel' in legacy:
            per_channel = bool(legacy.pop('per_channel'))
            alpha_group = 1 if per_channel else alpha_group
        # use_pw=True -> no_pw=False
        if 'use_pw' in legacy:
            use_pw_legacy = bool(legacy.pop('use_pw'))
            no_pw = (not use_pw_legacy)
        # ignore any other legacy keys silently (or log if you prefer)

        self.alpha_group = max(1, int(alpha_group))
        self.no_pw = bool(no_pw)
        self.use_bn = bool(use_bn)


        # ---------- α coefficients (group-shared) ----------
        self.alpha_group = max(1, int(alpha_group))
        G = max(1, self.C // self.alpha_group)   # number of channel groups
        ncoef = self.M + 1                       # center + M side taps
        # α stored per group → shape (G, ncoef)
        self.alpha = nn.Parameter(torch.zeros(G, ncoef))
        with torch.no_grad():
            self.alpha[:, 0].fill_(1.0)          # identity-safe init

        # ---------- optional channel mixing via PW (grouped) ----------
        self.use_bn = bool(use_bn)
        if not self.no_pw:
            H = max(1, self.C // max(1, int(pw_ratio)))
            # make groups legal for both 1x1 convs
            g = max(1, int(pw_groups))
            g = min(g, self.C, H)
            # ensure groups divide both C and H
            g = gcd(g, self.C)
            g = gcd(g, H) or 1
            self.pw_groups = g
            layers = [
                nn.Conv2d(self.C, H, 1, groups=g, bias=False),
                nn.BatchNorm2d(H) if self.use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(H, self.C, 1, groups=g, bias=False),
                nn.BatchNorm2d(self.C) if self.use_bn else nn.Identity(),
            ]
            self.pw = nn.Sequential(*layers)
        else:
            self.pw = nn.Identity()

        # ---------- global residual gate ----------
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    # build per-group 1D even kernel → expand to (C,1,K)
    def _build_even_kernel_1d(self, device, dtype):
        K = 2*self.M + 1
        G = max(1, self.C // self.alpha_group)
        # (G, K) then expand → (C, K)
        wg = torch.zeros(G, K, device=device, dtype=dtype)
        center = self.M
        wg[:, center] = self.alpha[:, 0]
        for m in range(1, self.M+1):
            val = self.alpha[:, m]
            wg[:, center - m] = val
            wg[:, center + m] = val if self.tie_sym else val  # kept symmetric here
        # normalize within each group (optional but stable)
        s = wg.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
        wg = wg / s
        # expand per group to channels
        reps = [self.alpha_group] * G
        reps[-1] = self.C - self.alpha_group*(G-1)  # handle remainder
        w = torch.cat([wg[i].unsqueeze(0).repeat(reps[i], 1) for i in range(G)], dim=0)  # (C, K)
        return w.unsqueeze(1)  # (C,1,K)

    def _pad(self, x, pad_h, pad_w):
        mode = 'reflect' if self.padding_mode == 'reflect' else \
               'replicate' if self.padding_mode == 'replicate' else 'constant'
        val = 0.0 if mode == 'constant' else None
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=mode, value=0.0 if val is not None else None)

    def forward(self, x):
        B, C, H, W = x.shape
        w1d = self._build_even_kernel_1d(x.device, x.dtype)  # (C,1,K)
        y = 0
        if 'h' in self.axis:
            wh = w1d.view(self.C, 1, 2*self.M+1, 1)
            xh = self._pad(x, pad_h=self.M*self.h, pad_w=0)
            yh = F.conv2d(xh, wh, stride=1, padding=0, dilation=(self.h,1), groups=self.C)
            y = y + yh
        if 'w' in self.axis:
            ww = w1d.view(self.C, 1, 1, 2*self.M+1)
            xw = self._pad(x, pad_h=0, pad_w=self.M*self.h)
            yw = F.conv2d(xw, ww, stride=1, padding=0, dilation=(1,self.h), groups=self.C)
            y = y + yw
        y = self.pw(y)
        return x + self.residual_scale * self.gate * y

# convenience alias for your previous import path
