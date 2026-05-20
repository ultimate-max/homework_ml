function y_uni = resample_log(logEntry, t_ref, fs)
%RESAMPLE_LOG 将不等间隔/变长仿真日志重采样到统一时间轴 t_ref。
%
% 修复 interp1 报错「X 和 V 的长度必须相同」：在插值前强制对齐 tv 与 yv。

    t_ref = t_ref(:);
    if nargin < 3 || isempty(fs)
        fs = 1000;
    end

    [tv, yv] = logentry_to_tv_yv(logEntry);

    if isempty(tv) || isempty(yv)
        warning('resample_log:EmptyLog', '日志为空，返回零序列。');
        y_uni = zeros(numel(t_ref), 1);
        return;
    end

    tv = tv(:);
    yv = yv(:);

    % 对齐长度（Simulink 偶发 time 与 data 差 1 点）
    n = min(numel(tv), numel(yv));
    if n < numel(tv) || n < numel(yv)
        warning('resample_log:LengthMismatch', ...
            '时间/数据长度不一致 (%d vs %d)，已截断为 %d。', ...
            numel(tv), numel(yv), n);
    end
    tv = tv(1:n);
    yv = yv(1:n);

    mask = isfinite(tv) & isfinite(yv);
    tv = tv(mask);
    yv = yv(mask);

    if numel(tv) < 2
        warning('resample_log:TooFewSamples', '有效样本 < 2，返回零序列。');
        y_uni = zeros(numel(t_ref), 1);
        return;
    end

    [tv, ia] = unique(tv, 'stable');
    yv = yv(ia);

    if numel(tv) < 2
        y_uni = zeros(numel(t_ref), 1);
        return;
    end

    t_uni = t_ref;
    if isempty(t_uni)
        t0 = tv(1);
        t1 = tv(end);
        dt = 1 / fs;
        t_uni = (t0:dt:t1).';
    end

    y_uni = interp1(tv, yv, t_uni, 'linear', 'extrap');
    y_uni = y_uni(:);
end


function [tv, yv] = logentry_to_tv_yv(logEntry)
    tv = [];
    yv = [];

    if isempty(logEntry)
        return;
    end

    if isa(logEntry, 'timeseries')
        tv = logEntry.Time(:);
        yv = logEntry.Data;
    elseif isstruct(logEntry) && isfield(logEntry, 'time') && isfield(logEntry, 'values')
        tv = logEntry.time(:);
        yv = logEntry.values;
    elseif isa(logEntry, 'Simulink.SimulationData.Signal')
        tv = logEntry.Values.Time(:);
        yv = logEntry.Values.Data;
    else
        return;
    end

    yv = squeeze(yv);
    if ismatrix(yv) && size(yv, 2) > 1
        % 单轴电机：取第一关节
        yv = yv(:, 1);
    end
    yv = yv(:);
end
