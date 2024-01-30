import torch
import torch.nn.functional as F
from utils.general import exists
from training.train_utils import do_kabsch


def kabsch_mse(
    pred,
    target,
    align_mask=None,
    mask=None,
    clamp=0.,
    sqrt=False,
    mseByResidue=False,
):  # 这里的align_mask就是batch_mask，对[b, n]产生一些mask
    aligned_target = do_kabsch(
        mobile=target,
        stationary=pred.detach(),
        align_mask=align_mask,
    )  # 优化得到旋转矩阵，将预测结构与已有结构进行对齐 (这里应该是对齐后的数据吧)
    mse = F.mse_loss(pred, aligned_target, reduction='none',).mean(-1)  # mse=1/n*sum((y_i-y'_i)**2)

    if clamp > 0:
        mse = torch.clamp(mse, max=clamp**2)

    if exists(mask):
        mse = torch.sum(
            mse * mask,
            dim=-1,
        ) / torch.sum(
            mask,
            dim=-1,
        )
    else:
        if not mseByResidue:
            mse = mse.mean(-1)

    if sqrt:
        mse = mse.sqrt()

    return mse
