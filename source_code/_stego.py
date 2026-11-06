"""Differentiable steganographic embedding with joint 27-ary changes of the RGB triplet.

Computes the change probabilities from costs under a payload constraint, simulates the embedding
(hard or Gumbel-softmax), and wraps Alice and Eve into StegoGAN27.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import torch
torch._dynamo.config.cache_size_limit = 16  # allowed versions per compiled function
torch._functorch.config.donated_buffer = False  # disable donated buffers
import torch.nn.functional as F
from types import SimpleNamespace
from typing import Tuple


# ========== PROJECTION MATRICES ==========
def get_deltas_ternary(q: int, device = torch.device('cpu'), return_index: bool = False, one_hot: bool = False) -> torch.Tensor:
    """Generate all combinations of q ternary digits (0, 1, 2), optionally mapped to (0, +1, -1).
    q (int): number of dimensions
    device (torch.device): target device
    return_index (bool): if True, return digits in {0,1,2}; if False, map {0,1,2} → {0,1,-1}
    returns: Tensor of shape (3^q, q)
    """
    base = 3
    # all numbers from 0 to 3^q - 1
    deltas = torch.arange(0, base ** q, device=device)
    # convert each number to base-3 representation
    digits = (deltas[:, None] // (base ** torch.arange(q, device=device))) % base  # 3^q, q
    if one_hot:
        digits = torch.tensor([
            [
                value
                for digit in row
                for value in (int(digit == 0), int(digit == 1), int(digit == 2))
            ]
            for row in digits
        ]).to(device)
        return digits
    if return_index:
        return digits
    # map {0, 1, 2} → {0, 1, -1}
    mapping = torch.tensor([0, 1, -1], device=device)
    deltas = mapping[digits]
    return deltas

H27 = SimpleNamespace()  # precomputed (module-level persistent) projection matrices
def initialize(device):
    # projection matrix for 27 -> 3 (values)
    H27._3 = get_deltas_ternary(3, device=device, return_index=False).float()  # 27 3
    H27._3 = H27._3[None, :, :, None, None]  # 1 27 3 1 1
    assert H27._3.shape == (1, 27, 3, 1, 1)
    # projection matrix for 27 -> 9 (one-hot), ordering [0, +1, -1]
    H27._3_onehot = (get_deltas_ternary(3, one_hot=True, device=device) != 0).float()  # 27 9
    H27._3_onehot = H27._3_onehot[None, :, :, None, None]  # 1 27 9 1 1
    assert H27._3_onehot.shape == (1, 27, 9, 1, 1)


# ========== BETA CONVERSIONS ==========
@torch.compile
def p3_sep_axes(p3: torch.Tensor) -> torch.Tensor:
    assert p3.shape[1] == 9
    B, C, H, W = p3.size()
    p3 = torch.stack([
        torch.stack([
            p3[:, 1],  # p_p1[R]
            p3[:, 2],  # p_m1[R]
        ], axis=1),
        torch.stack([
            p3[:, 4],  # p_p1[G]
            p3[:, 5],  # p_m1[G]
        ], axis=1),
        torch.stack([
            p3[:, 7],  # p_p1[B]
            p3[:, 8],  # p_m1[B]
        ], axis=1),
    ], axis=1)  # B 3 2 H W
    assert p3.shape == (B, 3, 2, H, W)
    return p3


@torch.compile
def p27_to_p3(p27: torch.Tensor, separate_axes: bool = False) -> torch.Tensor:
    """
    Convert 27-ary probability to per-channel additive probability of change.

    :param p27: Tensor of shape (B 27 H W), probability over all ternary combinations for 3 channels.
    :type p27: torch.Tensor
    :returns: Tensor of shape (B 9 H W), additive probability per channel of being changed. If separate_axes is True, the shape is (B 2 3 H W).
    :rtype: torch.Tensor
    """
    assert p27.size(1) == 27
    B, C, H, W = p27.shape
    # get change mask
    p27 = p27[:, :, None]  # B 27 1 H W
    p3 = (p27 * H27._3_onehot).sum(dim=1)  # B 9 H W
    assert p3.shape == (B, 9, H, W)
    #
    if separate_axes:
        p3 = p3_sep_axes(p3)  # B 3 2 H W
    return p3


@torch.compile
def p3_to_p2(p3: torch.Tensor) -> torch.Tensor:
    if p3.size(1) == 9:
        p3 = p3_sep_axes(p3)  # B 3 2 H W
    return torch.sum(p3, dim=2)  # B 3 H W

@torch.compile
def p1_to_p2(p1: torch.Tensor) -> torch.Tensor:
    return p1[:, 1:2] + p1[:, 2:3]  # B 1 H W

@torch.compile
def p27_to_p2(p27: torch.Tensor) -> torch.Tensor:
    """
    Convert 27-ary probability to per-channel additive probability of change.

    :param p27: tensor of shape (B 27 H W), probability over all ternary combinations for 3 channels.
    :type p27: torch.Tensor
    :returns: tensor of shape (B 3 H W), additive probability per channel of being changed.
    :rtype: torch.Tensor
    """
    assert p27.size(1) == 27
    B, C, H, W = p27.size()
    p3 = p27_to_p3(p27)  # B 9 H W
    assert p3.shape == (B, 9, H, W)
    p2 = p3_to_p2(p3)  # B 3 H W
    assert p2.shape == (B, 3, H, W)
    return p2


@torch.compile
def p3_to_p27(p3: torch.Tensor) -> torch.Tensor:
    assert p3.size(1) == 9
    p3 = p3.unsqueeze(1)
    p27 = torch.where(H27._3_onehot.bool(), p3, torch.ones_like(p3)).prod(dim=2)  # B 27 H W
    return p27


# ========== ADJUSTING ==========
@torch.compile
def adjust_rho27(rho27: torch.Tensor, x0: torch.Tensor, wet_cost: float = 10**10) -> torch.Tensor:
    # replace inf/nan/>wet
    rho27 = torch.where(
        torch.isfinite(rho27) & (rho27 < wet_cost),
        rho27,
        torch.tensor(wet_cost, device=rho27.device, dtype=rho27.dtype),
    )  # B 27 H W

    # boundaries (1/255 = .00392)
    at_max = (x0 >= .999).unsqueeze(1)  # B 1 3 H W
    at_min = (x0 <= .001).unsqueeze(1)  # B 1 3 H W
    # violation
    violation = (at_max & (H27._3 > 0)) | (at_min & (H27._3 < 0))  # B 27 3 H W
    mask27 = violation.any(dim=2)  # B 27 H W
    # masking
    rho27 = torch.where(mask27, torch.tensor(wet_cost, device=rho27.device), rho27)  # B 27 H W
    return rho27  # B 27 H W

@torch.compile
def adjust_rho1(rho1: torch.Tensor, x0: torch.Tensor, wet_cost: float = 10**10) -> torch.Tensor:
    """adjusts rho1 according to the saturated areas in x0.

    rho1 has a shape (B, 3, H, W), where the channels carry costs ordered as (rho_0, rho_p1, rho_m1).
    x0 has a shape (B, 1, H, W)
    """
    # replace inf/nan/>wet
    rho1 = torch.where(
        torch.isfinite(rho1) & (rho1 < wet_cost),
        rho1,
        torch.tensor(wet_cost, device=rho1.device, dtype=rho1.dtype),
    )  # B 3 H W

    # boundaries (1/255 = .00392)
    mask = torch.zeros(rho1.shape, dtype=torch.bool, device=rho1.device)
    mask[:, 1:2] = (x0 >= .999)  # B 1 H W
    mask[:, 2:3] = (x0 <= .001)  # B 1 H W
    # masking
    rho1 = torch.where(mask, torch.tensor(wet_cost, device=rho1.device), rho1)  # B 3 H W
    return rho1  # B 3 H W

@torch.compile
def adjust_rho3(rho3: torch.Tensor, x0: torch.Tensor, wet_cost: float = 10e6) -> torch.Tensor:
    # replace inf/nan/>wet
    rho3 = torch.where(
        torch.isfinite(rho3) & (rho3 < wet_cost),
        rho3,
        torch.full_like(rho3, wet_cost)
    )  # B 9 H W

    # boundaries (1/255 = .00392)
    wet3_p1 = (x0 >= .999)  # B 3 H W
    wet3_m1 = (x0 <= .001)  # B 3 H W
    #
    mask3 = torch.stack([
        torch.zeros_like(wet3_p1[:, 0]), wet3_p1[:, 0], wet3_m1[:, 0],
        torch.zeros_like(wet3_p1[:, 1]), wet3_p1[:, 1], wet3_m1[:, 1],
        torch.zeros_like(wet3_p1[:, 2]), wet3_p1[:, 2], wet3_m1[:, 2],
    ], dim=1)  # B 9 H W
    rho3 = torch.where(mask3, torch.tensor(wet_cost, device=x0.device), rho3)
    #
    return rho3  # B 9 H W

# ========== ENTROPY CONSTRAINT ==========
@torch.compile
def qary_entropy(p: torch.Tensor, dim=None) -> torch.Tensor:
    assert p.size(1) in {2, 3, 6, 9, 27}, 'strange q-arity encountered'
    #
    eps = torch.finfo(p.dtype).eps
    p = torch.clamp(p, eps, 1-eps)  # B 27 H W
    #
    if dim is None:
        dim = tuple(range(1, p.ndim))
    h_hat = -torch.sum(p * torch.log2(p), dim=dim)
    return h_hat


@torch.compile
def calc_lambda(rho, m, objective, xtol=1e-10, max_iter=100, min_value=1e-30):
    """Lambda search via binary search, to be called per batch."""
    device = rho.device
    B = rho.shape[0]
    if isinstance(m, (int, float)):
        m = torch.full((B,), m, dtype=rho.dtype, device=device)
    # bracket search
    lbd_l = torch.full((B,), min_value, dtype=rho.dtype, device=device)
    lbd_r = torch.full((B,), min_value, dtype=rho.dtype, device=device)
    active_mask = torch.ones(B, dtype=torch.bool, device=device)
    for it in range(50): # 1e-30 to 1e16 in 10x steps is ~46 iterations
        _, v = objective(rho, lbd_r.view(-1, 1, 1, 1))  # current payload for active images
        still_too_high = (v >= m) & (lbd_r < 1e16)  # check which images remain active
        lbd_l = torch.where(still_too_high, lbd_r, lbd_l)  # shift lower bound
        lbd_r = torch.where(still_too_high, lbd_r * 10, lbd_r)  # multiply upper bound
        active_mask = still_too_high  # carry the active images
        if not active_mask.any():
            break
    # bisection
    for it in range(max_iter):  # log2((max-min)/xtol) steps needed (70 is enough for f32 and 1e-3)
        mid = (lbd_l + lbd_r) / 2  # middle
        _, v = objective(rho, mid.view(-1, 1, 1, 1))  # current payload
        too_high = v > m  # splitting rule
        lbd_l = torch.where(too_high, mid, lbd_l)  # lower half
        lbd_r = torch.where(~too_high, mid, lbd_r)  # upper half
        if (lbd_r - lbd_l).max() < xtol:  # early exit
            break
    #
    lbd = (lbd_l + lbd_r) / 2  # midpoint is the final lambda
    _, h_hat = objective(rho, lbd.view(-1, 1, 1, 1))  # final entropy
    return lbd, h_hat


@torch.compile
def average_payload(
    rho: torch.Tensor = None,
    lbda: float = None,
) -> Tuple[torch.Tensor, float]:
    """Objective: code-native entropy."""
    p = F.softmax(-lbda * rho, dim=1)  # get selection channel
    h_hat = qary_entropy(p)
    return p, h_hat


@torch.compile
def average_payload_27to3(
    rho: torch.Tensor = None,
    lbda: float = None,
) -> Tuple[torch.Tensor, float]:
    """Objective: entropy of the (projected) 3-ary model."""
    assert rho.size(1) == 27
    #
    p27, h27_hat = average_payload(rho=rho, lbda=lbda)
    p3 = p27_to_p3(p27)  # B 9 ...
    h3_hat = qary_entropy(p3)
    return p27, h3_hat


class find_lambda(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    xtol = 1e-10
    max_iter = 30
    min_value = 1e-30
    @classmethod
    def forward(cls, ctx, rho, m, n, objective):
        B = rho.size(0)
        # bracket + bisection search for lambda
        lbdas, h_hat = calc_lambda(
            rho=rho,
            m=m,
            objective=objective,
            xtol=cls.xtol,
            max_iter=cls.max_iter,
            min_value=cls.min_value,
        )
        lbdas = lbdas.view(B, 1, 1, 1)
        with torch.enable_grad():
            r = rho.detach().requires_grad_(True)
            l = lbdas.detach().requires_grad_(True)

        ctx.save_for_backward(r, l)
        ctx.objective = objective
        return lbdas.detach().clone().requires_grad_(True)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.enable_grad():
            rho, lbda = ctx.saved_tensors  # 1 9 H W
            _, h_hat = ctx.objective(rho, lbda)

            # Gradients
            grad_h_lbda = torch.autograd.grad(h_hat, lbda, torch.ones_like(h_hat), retain_graph=True)[0]
            grad_h_rho = torch.autograd.grad(h_hat, rho, torch.ones_like(h_hat), retain_graph=False)[0]

            # Implicit gradient
            g = -grad_h_rho / (grad_h_lbda - 1e-9)
        return grad_output * g, None, None, None


# ========== SIMULATION ==========
@torch.compile
def _simulate(
    p: torch.Tensor,
    rand: torch.Tensor,  # B ~ U(0,1), B 1 H W
    simulator_method: str = 'hard',
) -> torch.Tensor:
    """Simulates ternary steganographic embedding via MI according to the distributions p."""
    B, C, H, W = p.size()
    p_cum = torch.cumsum(p, dim=1)
    #
    indices = torch.argmax((rand < p_cum).to(torch.int8), dim=1)  # B H W
    delta3_hard = F.one_hot(indices, num_classes=C).permute(0, 3, 1, 2).to(p.dtype)
    delta3_hard = torch.round(delta3_hard)
    if simulator_method == 'hard':
        delta3 = delta3_hard
    elif simulator_method == 'probability-ste':
        delta3 = (delta3_hard - p).detach() + p  # project p to p3
    else:
        raise NotImplementedError(f'unknown simulator method {simulator_method}')
    return delta3

@torch.compile
def _simulate_differentiable(
    p: torch.Tensor,  # B 27 H W, per-pixel categorical probabilities
    rand: torch.Tensor,  # B ~ U(0,1), B 27 H W
    tau: float = 1.0,  # temperature for Gumbel-Softmax (1.0 is default)
    simulator_method: str = 'gumbel-softmax',
) -> torch.Tensor:
    """
    Differentiable q-ary sampling with Straight-Through Estimator (hard Gumbel-Softmax).

    Forward pass: hard categorical selection -> real delta pattern
    Backward pass: gradient flows through soft expectation
    """
    # Gumbel noise for sampling
    eps = 1e-20
    gumbel_noise = -torch.log(-torch.log(rand + eps) + eps)

    # Soft sample
    logits = torch.log(p + eps)  # avoid log(0)
    delta_soft = torch.softmax((logits + gumbel_noise) / tau, dim=1)  # B 27 H W

    # Hard categorical sample (forward)
    idx = delta_soft.argmax(dim=1, keepdim=True)  # B 1 H W
    delta_hard = torch.zeros_like(p).scatter_(1, idx, 1.0)

    # Straight-through: forward = hard, backward = soft
    if simulator_method == 'gumbel-softmax':
        delta3 = (delta_hard - delta_soft).detach() + delta_soft  # B 27 H W
    elif simulator_method == 'vanilla':
        delta3 = delta_soft
    else:
        raise NotImplementedError(f'unknown simulator method {simulator_method}')
    return delta3


def generate_randomness(shape: Tuple[int], stego_seed: int, device: torch.device):
    """Generates randomness."""
    # reproducibility
    rng = torch.Generator(device=device)
    if stego_seed is not None:
        rng.manual_seed(stego_seed)
    # randomness
    rand = torch.rand(shape, generator=rng, device=device)
    return rand


def simulate(
    p: torch.Tensor,          # B C H W, per-pixel categorical probabilities
    deltas: torch.Tensor = None,  # q^3 q, delta patterns corresponding to each category
    tau: float = 1.0,  # temperature for Gumbel-Softmax (1.0 is default)
    simulator_method: str = 'hard',  # expectation, gumbel-softmax, vanilla, probability-ste, hard
    stego_seed: int = None,
) -> torch.Tensor:
    """
    Differentiable q-ary sampling with Straight-Through Estimator (hard Gumbel-Softmax).

    Forward pass: hard categorical selection -> real delta pattern
    Backward pass: gradient flows through soft expectation
    """
    device = p.device
    B, C, H, W = p.shape
    if C == 27 and deltas is None:
        deltas = H27._3
    # --------------------------------------
    # # simulate
    if simulator_method in {'hard', 'probability-ste'}:
        rand = generate_randomness((B, 1, H, W), stego_seed=stego_seed, device=device)
        delta = _simulate(
            p=p,
            rand=rand,
            simulator_method=simulator_method,
        )  # B C H W
    elif simulator_method in {'gumbel-softmax', 'vanilla'}:
        rand = generate_randomness((B, C, H, W), stego_seed=stego_seed, device=device)
        delta = _simulate_differentiable(
            p=p,
            rand=rand,
            tau=tau,
            simulator_method=simulator_method,
        )  # B C H W
    else:
        delta = p  # B C H W

    # --------------------------------------
    # Map categorical index to delta pattern
    assert delta.shape == (B, C, H, W)
    if deltas is not None:
        delta3 = torch.einsum('bchw, ncqxy -> bqhw', delta, deltas)  # B 3 H W
    else:
        delta3 = delta
    return delta3, delta


# ========== GAN ==========
class StegoGAN27(torch.nn.Module):
    def __init__(
        self,
        net_a: torch.nn.Module,
        net_e: torch.nn.Module = None,
        pad: int = None,
    ):
        super().__init__()
        self.net_a = net_a
        self.net_e = net_e
        self.pad = pad


    def get_probabilities(
        self,
        x0: torch.Tensor,
        alpha: torch.Tensor,
        tau: float = 1.,
        mode: bool = 'a',
    ):
        """Alice' pass: differentiable calculation of the selection channel."""
        context = torch.enable_grad() if mode == 'a' else torch.no_grad()
        with context:
            B, C, H, W = x0.size()
            assert C == 3
            N = C * H * W
            # pad
            if mode == 'a': x0.requires_grad_(True)  # capture gradients
            x0_ = F.pad(x0, (self.pad, self.pad, self.pad, self.pad), mode='reflect') if self.pad else x0
            # cost
            rho27 = self.net_a(x0_)
            # unpad
            rho27 = rho27[..., self.pad:H+self.pad, self.pad:W+self.pad] if self.pad else rho27
            rho27 = adjust_rho27(rho27, x0)  # adjust
            # selection channel
            objective = average_payload_27to3  # average_payload_27to3 average_payload
            lbda = find_lambda.apply(rho27, alpha * N, N, objective)
            p27, _ = objective(rho27, lbda)
            p3 = p27_to_p3(p27)  # B 9 H W
            p2 = p27_to_p2(p27)  # B 3 H W
            h27, h3 = qary_entropy(p27), qary_entropy(p3)
            alpha3_ch = qary_entropy(p3.view(B*C, 3, H, W)).view(B, C) / H / W  # B C
            beta2 = torch.mean(torch.sum(p2, dim=1, keepdim=True), dim=(1, 2, 3))  # B
            beta2_ch = torch.mean(p2, dim=(2, 3))  # B C
            # effective selection channel (accounting for temperature)
            p27e, _ = objective(rho27, lbda / tau)
            p3e = p27_to_p3(p27e)
            h27e, h3e = qary_entropy(p27e), qary_entropy(p3e)
            alpha3e_ch = qary_entropy(p3e.view(B*C, 3, H, W)).view(B, C) / H / W  # B C
        #
        if mode == 'a':
            x0.retain_grad()
            rho27.retain_grad()
            lbda.retain_grad()
            p27.retain_grad()
        #
        return SimpleNamespace(
            rho27=rho27,
            lbda=lbda,
            p27=p27, h27=h27, alpha27=h27 / N,
            p3=p3, h3=h3, alpha3=h3 / N, alpha3_ch=alpha3_ch,
            p2=p2, beta2=beta2, beta2_ch=beta2_ch,
            p27e=p27e, h27e=h27e, alpha27e=h27e / N,
            p3e=p3e, h3e=h3e, alpha3e=h3e / N, alpha3e_ch=alpha3e_ch,
            x0=x0)

    def simulate(
        self,
        x0: torch.Tensor,
        p27: torch.Tensor,
        tau: float = 1.,
        stego_seed: int = None,
        simulator_method: str = 'gumbel-softmax',
        mode: bool = 'a',
    ):
        """Alice' pass: differentiable simulator."""
        context = torch.enable_grad() if mode == 'a' else torch.no_grad()
        with context:
            delta3, delta27 = simulate(
                p=p27,  # expand draws
                tau=tau,
                simulator_method=simulator_method,
                stego_seed=stego_seed,
            )
            delta3 = delta3.to(x0.dtype)
            delta27 = delta27.to(x0.dtype)
            x1 = x0 + delta3 / 255.
        #
        if mode == 'a':
            x1.retain_grad()
        #
        return SimpleNamespace(
            delta27=delta27,
            delta3=delta3,
            x1=x1,
            x0=x0)

    def discriminate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        mode: bool = 'a'
    ):
        """Eve's pass."""
        if self.net_e is None:
            return SimpleNamespace(l0=None, l1=None, s0=None, s1=None)
        # D forward
        context = torch.enable_grad() if mode in {'a', 'e'} else torch.no_grad()
        with context:
            l0 = self.net_e(x0)
            l1 = self.net_e(x1)
            # activation
            s0, s1 = F.softmax(l0, dim=1), F.softmax(l1, dim=1)
        return SimpleNamespace(l0=l0, l1=l1, s0=s0, s1=s1)  # f0=f0, f1=f1,

    def forward(
        self,
        x0: torch.Tensor,
        alpha: torch.Tensor,
        simulator_method: str,
        stego_seed: int,
        no_eve: bool = False,
        tau: float = 1.,
        mode: bool = 'pass',  # a, e, pass
    ):
        """Wrapper for forward."""
        # 1. Decision
        res_p = self.get_probabilities(x0, alpha, tau=tau, mode=mode)
        # 2. Action
        res_s = self.simulate(
            x0, res_p.p27, tau=tau,
            stego_seed=stego_seed,
            simulator_method=simulator_method,
            mode=mode)
        # 3. Judgment
        if no_eve:
            res_d = SimpleNamespace(l0=None, l1=None, s0=None, s1=None)
        else:
            res_d = self.discriminate(x0, res_s.x1, mode=mode)
        # Merge all namespaces
        delattr(res_s, 'x0')
        return SimpleNamespace(**vars(res_p), **vars(res_s), **vars(res_d))

