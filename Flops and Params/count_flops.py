import torch
from fvcore.nn import FlopCountAnalysis
import pandas as pd


model = Model(n_channels=1, n_classes=9).cuda()
dummy_input = torch.randn(1, 1, 224, 224).cuda()

flops = FlopCountAnalysis(model, dummy_input)

total_flops = flops.total()  # 返回int，单位是 FLOPs（比如：数量级为10^9）
gflops = total_flops / 1e9   # 换算成 GFlops

print(f"Total FLOPs (float-safe): {float(total_flops):.0f} ({gflops:.3f} GFlops)")


# 每个模块的 FLOPs，转换成 dict，方便处理
flops_dict = flops.by_module()

# 将 dict 转成列表（模块名称，FLOPs，GFlops）
data = [(str(k), v, v / 1e9) for k, v in flops_dict.items()]

# 创建 DataFrame
df = pd.DataFrame(data, columns=["Module", "FLOPs", "GFlops"])

# 按 FLOPs 降序排序，方便查看大模块
df = df.sort_values(by="FLOPs", ascending=False)

# 打印表格
print(df.to_string(index=False))


