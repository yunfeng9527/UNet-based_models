def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 单位：百万（M）
    total_params_M = total_params / 1e6
    trainable_params_M = trainable_params / 1e6

    # 单位：MB（假设 float32，每参数占 4 字节）
    total_size_MB = total_params * 4 / (1024 ** 2)
    trainable_size_MB = trainable_params * 4 / (1024 ** 2)

    return {
        'Total Params (M)': round(total_params_M, 2),
        'Trainable Params (M)': round(trainable_params_M, 2),
        'Total Size (MB)': round(total_size_MB, 2),
        'Trainable Size (MB)': round(trainable_size_MB, 2)
    }