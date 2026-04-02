import torch
from torch.optim.optimizer import Optimizer
import math


class AdaBelief(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-16,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        decoupled_decay: bool = True,
        fixed_decay: bool = False,
        rectify: bool = False,
        degenerated_to_sgd: bool = True,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad,
            decoupled_decay=decoupled_decay,
            fixed_decay=fixed_decay, rectify=rectify,
            degenerated_to_sgd=degenerated_to_sgd,
        )
        super(AdaBelief, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients")

                amsgrad = group['amsgrad']
                state = self.state[p]
                beta1, beta2 = group['betas']

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient (m_t, Eq. 28)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of (g_t - m_t)^2 (v_t, Eq. 29)
                    state['exp_avg_var'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_var'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decoupled weight decay
                if group['decoupled_decay']:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                elif group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Update m_t (Eq. 28): m_t = β1 * m_{t-1} + (1 - β1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update v_t (Eq. 29): v_t = β2 * v_{t-1} + (1-β2)*(g_t - m_t)^2 + ε
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(
                    grad_residual, grad_residual, value=1 - beta2
                ).add_(group['eps'])  # Add eps for numerical stability

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                    denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # Bias-corrected first moment
                step_size = group['lr'] / bias_correction1

                # Update θ_t (Eq. 30): θ_t = θ_{t-1} - η * m_t / (sqrt(v_t) + ε)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class CosineAnnealingWithCycles:
    def __init__(self,optimizer: Optimizer,T_max: int,n_cycles: int = 5,eta_min_ratio: float = 0.01,):

        self.optimizer = optimizer
        self.T_max = T_max
        self.n_cycles = n_cycles
        self.eta_min_ratio = eta_min_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0

    def get_lr(self, step: int) -> list:
        lrs = []
        for base_lr in self.base_lrs:
            eta_min = base_lr * self.eta_min_ratio
            # t in [0, T_max], with n_cycles complete cycles
            T_cycle = max(1, self.T_max // self.n_cycles)
            t = step % T_cycle
            # Eq. 31: η_t = η_min + (η_max - η_min) * (1 + cos(π*t/T)) / 2
            lr = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * t / T_cycle)) / 2
            lrs.append(lr)
        return lrs

    def step(self):
        self.current_step += 1
        lrs = self.get_lr(self.current_step)
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'base_lrs': self.base_lrs,
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.base_lrs = state_dict['base_lrs']
