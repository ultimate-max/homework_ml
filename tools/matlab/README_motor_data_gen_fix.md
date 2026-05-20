# motor_data_gen：`interp1` 长度不一致修复

## 原因

`resample_log` 里 `interp1(tv, yv, ...)` 要求 **`numel(tv) == numel(yv)`**。Simulink 日志常见情况：

1. `Values.Data` 为 `1×T` 却被 `(:)` 拉成长度 `T` 的 `yv`，而 `Time` 为 `T×1`（或反过来）
2. `time` 用 `simOut.tout`，`values` 用 `logsout` 里另一套采样，**长度差 1**
3. 多自由度矩阵未按行对齐，误把 `T×n` 展平成 `T*n`

## 用法 A（推荐）：改用本目录工具函数

在 MATLAB 中：

```matlab
addpath('/home/coral/project/DeLaN_Stribeck/tools/matlab');
```

在 **`motor_data_gen.m` 中删除**（或注释掉）本地的 `get_log_var`、`resample_log` 子函数，改为调用上述两个文件。

`collect_logged_outputs` 保持：

```matlab
qd = resample_log(get_log_var(simOut, 'qd'), t_ref, fs);
```

## 用法 B：只改 `resample_log` 子函数

在现有 `motor_data_gen.m` 的 `resample_log` 里，在 `interp1` **之前**加入：

```matlab
tv = tv(:);
yv = yv(:);
n = min(numel(tv), numel(yv));
if n < numel(tv) || n < numel(yv)
    warning('resample_log: 截断 tv(%d) yv(%d) -> %d', numel(tv), numel(yv), n);
end
tv = tv(1:n);
yv = yv(1:n);
mask = isfinite(tv) & isfinite(yv);
tv = tv(mask);
yv = yv(mask);
[tv, ia] = unique(tv, 'stable');
yv = yv(ia);
```

并确认 `yv` 来自 **与 `tv` 同行** 的数据（单轴时 `values(:,1)`，不要用 `values(:)` 除非 `values` 已是列向量）。

## 调试

在 `collect_logged_outputs` 里临时加：

```matlab
e = get_log_var(simOut, 'qd');
fprintf('qd: numel(time)=%d, size(values)=%s\n', numel(e.time), mat2str(size(e.values)));
```

若 `size(values,1) ~= numel(time)`，问题在 `get_log_var` 的取向/维度假设。
