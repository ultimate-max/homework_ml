function logEntry = get_log_var(simOut, name)
%GET_LOG_VAR 从 Simulink 仿真结果中取出带时间轴的日志（供 resample_log 使用）。
%
% 返回 struct: .time (T×1), .values (T×1) 或 (T×n_dof)，且 length(time)==size(values,1)。

    name = char(name);
    logEntry = struct('time', [], 'values', []);

    if nargin < 1 || isempty(simOut)
        return;
    end

    % 1) logsout（R2016b+ Dataset）
    if isprop(simOut, 'logsout') && ~isempty(simOut.logsout)
        try
            elem = simOut.logsout.get(name);
            if ~isempty(elem)
                logEntry = signal_to_logentry(elem);
                if ~isempty(logEntry.time)
                    return;
                end
            end
        catch
        end
        try
            elem = simOut.logsout.getElement(name);
            if ~isempty(elem)
                logEntry = signal_to_logentry(elem);
                if ~isempty(logEntry.time)
                    return;
                end
            end
        catch
        end
    end

    % 2) 顶层字段 simOut.qd 等
    if isprop(simOut, name)
        logEntry = signal_to_logentry(simOut.(name));
        if ~isempty(logEntry.time)
            return;
        end
    end

    % 3) Simulink.SimulationOutput 的 get()
    try
        v = simOut.get(name);
        logEntry = signal_to_logentry(v);
    catch
    end
end


function logEntry = signal_to_logentry(sig)
    logEntry = struct('time', [], 'values', []);

    if isempty(sig)
        return;
    end

    if isa(sig, 'timeseries')
        tv = sig.Time(:);
        yv = sig.Data;
    elseif isa(sig, 'Simulink.SimulationData.Signal')
        tv = sig.Values.Time(:);
        yv = sig.Values.Data;
    elseif isstruct(sig)
        if isfield(sig, 'time')
            tv = sig.time(:);
            if isfield(sig, 'values')
                yv = sig.values;
            elseif isfield(sig, 'signals') && isfield(sig.signals, 'values')
                yv = sig.signals.values;
            else
                yv = [];
            end
        elseif isfield(sig, 'Time')
            tv = sig.Time(:);
            if isfield(sig, 'Data')
                yv = sig.Data;
            else
                yv = [];
            end
        else
            tv = [];
            yv = [];
        end
    else
        tv = [];
        yv = [];
    end

    if isempty(tv) || isempty(yv)
        return;
    end

    yv = squeeze(yv);
    if isvector(yv)
        yv = yv(:);
    else
        % 约定：时间在行上 (T×n_dof)
        if size(yv, 1) ~= numel(tv) && size(yv, 2) == numel(tv)
            yv = yv.';
        end
        if size(yv, 1) ~= numel(tv)
            % 无法对齐则取第一列并截断
            yv = yv(:, 1);
        end
    end

    n = min(numel(tv), size(yv, 1));
    logEntry.time = tv(1:n);
    logEntry.values = yv(1:n, :);
end
