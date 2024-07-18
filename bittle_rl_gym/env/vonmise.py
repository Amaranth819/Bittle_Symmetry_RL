'''
    Pytorch implementation of Vonmise distribution.

    https://pytorch.org/docs/stable/special.html
    https://math.stackexchange.com/questions/1255902/why-is-the-area-under-the-pdf-for-the-von-mises-distribution-not-one

    However, it seems like Pytorch is much more computationally expensive than scipy here!
'''

import matplotlib.pyplot as plt
import scipy.special
import torch
import numpy as np
from scipy.stats import vonmises, vonmises_line


def vonmise_pdf(x, kappa, loc = 0):
    numerator = torch.exp(kappa * torch.cos(x - loc))
    denominator = 2 * torch.pi * torch.special.i0(kappa)
    return numerator / denominator




def test_vonmise_pdf():
    from scipy.stats import vonmises
    success, total = 0, 10000
    errors = []

    for _ in range(total):
        x = (2 * torch.rand(size = (1,)) - 1) * torch.pi
        loc = (2 * torch.rand(size = (1,)) - 1) * torch.pi
        kappa = torch.rand(size = (1,)) * torch.randint(low = 0, high = 10, size = (1,)) + 1e-7

        pdf_pt = vonmise_pdf(x, loc, kappa).numpy()
        pdf_np = vonmises.pdf(x = x.numpy(), loc = loc.numpy(), kappa = kappa.numpy())
        error = np.abs(pdf_pt - pdf_np)
        errors.append(error)
        if error <= 1e-5:
            success += 1

    print(success, total, success / total)
    print(np.mean(errors))



def plot_vonmise_pdf():
    x = torch.linspace(-4 * torch.pi, 4 * torch.pi, 500)
    loc = torch.ones(1) * 0
    kappa = torch.ones(1)
    
    y = vonmise_pdf(x, kappa, loc)
    plt.plot(x.numpy(), y.numpy())
    y = vonmises.pdf(x = x.numpy(), kappa = kappa.numpy(), loc = loc.numpy())
    plt.plot(x.numpy(), y)
    plt.savefig('vonmise_pdf.png')
    plt.close()


def von_mises_cdf_series(k, x, p):
    s, c = torch.sin(x), torch.cos(x)
    sn, cn = torch.sin(p * x), torch.cos(p * x)
    R, V = torch.zeros_like(k), torch.zeros_like(k)
    
    for n in range(p - 1, 0, -1):
        sn, cn = sn * c - cn * s, cn * c + sn * s
        R = k / (2 * n + k * R)
        V = R * (sn / n + V)

    return 0.5 + (0.5 * x + V) / torch.pi


def von_mises_cdf_normalapprox(k, x):
    SQRT2_PI = 0.79788456080286535588

    b = SQRT2_PI / torch.special.i0(k)  # Check for negative k
    z = b * torch.sin(0.5 * x)
    return torch.special.ndtr(z)


def vonmise_cdf(x, kappa, loc = 0):
    # https://github.com/scipy/scipy/blob/941d5b08614841a213019990b6ceee83a05d6dcc/scipy/stats/_stats.pyx#L46
    x = x - loc
    ix = torch.round(0.5 * x / torch.pi)
    x = x - 2 * torch.pi * ix

    CK = 50
    a1, a2, a3, a4 = 28., 0.5, 100., 5.

    # bx, bk = torch.broadcast_tensors(x, kappa)
    bx, bk = x, kappa
    result = torch.empty_like(bx, dtype = torch.float)

    c_small_k = bk < CK
    temp = result[c_small_k]
    temp_xs = bx[c_small_k].float()
    temp_ks = bk[c_small_k].float()
    # for i in range(temp.size(0)):
    #     p = (1 + a1 + a2 * temp_ks[i] - a3 / (temp_ks[i] + a4)).int()
    #     temp[i] = von_mises_cdf_series(temp_ks[i], temp_xs[i], p)
    #     temp[i] = torch.clamp(temp[i], 0, 1)

    p = (1 + a1 + a2 * temp_ks - a3 / (temp_ks + a4)).int()
    temp[:] = torch.clamp(von_mises_cdf_series(temp_ks, temp_xs, p), 0, 1)

    result[c_small_k] = temp
    result[~c_small_k] = von_mises_cdf_normalapprox(bk[~c_small_k], bx[~c_small_k])

    return result + ix


if __name__ == '__main__':
    import time

    # r = 3
    # x = torch.linspace(-torch.pi*r, torch.pi*r, 4096).cuda()
    # kappa = 1 * torch.ones(4096).cuda()
    # start = time.time()
    # y1 = vonmise_cdf(x, kappa, loc = torch.pi).cpu().numpy()
    # print(time.time() - start)
    # plt.plot(x.cpu().numpy(), y1, color = 'blue', label = 'pytorch')

    # from scipy.stats import vonmises, vonmises_line
    # start = time.time()
    # x = x.cpu().numpy()
    # kappa = kappa.cpu().numpy()
    # y2 = vonmises.cdf(x, kappa = kappa, loc = np.pi)
    # print(time.time() - start)
    # plt.plot(x, y2, '--', color = 'red', label = 'numpy')

    # start = time.time()
    # y3 = vonmises_line.cdf(x, kappa = kappa, loc = np.pi)
    # print(time.time() - start)
    # plt.plot(x, y3, '--', color = 'green', label = 'numpy vonmise_line')

    # plt.legend()
    # plt.savefig('vonmise_cdf.png')
    # plt.close()
    
    # print(np.mean(y1 - y2))
    # print(np.mean(y1 - y3))





    # x = torch.ones(4096).cuda()
    # start = time.time()
    # torch.special.i0(x)
    # print(time.time() - start)

    # import scipy
    # start = time.time()
    # torch.from_numpy(scipy.special.i0(np.ones(4096))).cuda()
    # print(time.time() - start)

    d = 6
    num_envs = 4096
    x = torch.ones(size = (num_envs, d)).cuda()
    kappa = torch.ones(size = (num_envs, d)).cuda() + 1e-7


    start = time.time()
    P1 = vonmises_line.cdf(x = x[..., 0].cpu().numpy(), kappa = kappa[..., 0].cpu().numpy())
    P2 = vonmises_line.cdf(x = x[..., 1].cpu().numpy(), kappa = kappa[..., 1].cpu().numpy())
    P3 = vonmises_line.cdf(x = x[..., 2].cpu().numpy(), kappa = kappa[..., 2].cpu().numpy())
    P4 = vonmises_line.cdf(x = x[..., 3].cpu().numpy(), kappa = kappa[..., 3].cpu().numpy())
    P5 = vonmises_line.cdf(x = x[..., 4].cpu().numpy(), kappa = kappa[..., 4].cpu().numpy())
    P6 = vonmises_line.cdf(x = x[..., 5].cpu().numpy(), kappa = kappa[..., 5].cpu().numpy())
    res1 = P1 * (1 - P2) + P3 * (1 - P4) + P5 * (1 - P6)
    print(time.time() - start)

    start = time.time()
    Ps = vonmises_line.cdf(x = x.cpu().numpy(), kappa = kappa.cpu().numpy())
    res2 = Ps[..., 0] * (1 - Ps[..., 1]) + Ps[..., 2] * (1 - Ps[..., 3]) + Ps[..., 4] * (1 - Ps[..., 5])
    print(time.time() - start)

    kappa = torch.ones(size = (num_envs, )).cuda() + 1e-7
    start = time.time()
    Ps = vonmises_line.cdf(x = x.cpu().numpy(), kappa = kappa.unsqueeze(-1).cpu().numpy())
    res3 = Ps[..., 0] * (1 - Ps[..., 1]) + Ps[..., 2] * (1 - Ps[..., 3]) + Ps[..., 4] * (1 - Ps[..., 5])
    print(time.time() - start)

    print(np.linalg.norm(res1 - res2))
    print(np.linalg.norm(res1 - res3))
    print(np.linalg.norm(res2 - res3))