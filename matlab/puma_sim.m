%% PUMA560 数据集：每字符一条五阶傅里叶激励轨迹（建模数据集式，式 29）
% q_k(t)=q_{k,0}+sum_{i=1}^5 (a_{k,i}/(i*omega_f)*sin(i*omega_f*t) - b_{k,i}/(i*omega_f)*cos(i*omega_f*t))
% omega_f 取 0.1 Hz；a_{k,i}, b_{k,i} 由迭代优化在关节限位内最大化激励度
% qd、qdd 解析求导；末端仅 fkine 用于可视化
% 关节摩擦：Stribeck（式 15–16）；标称 + 按标签初值 + 轨迹内缓慢漂移
% 依赖：Peter Corke Robotics Toolbox（mdl_puma560、rne、fkine）
%
% 输出：character_data.mat（与脚本同目录）

clear; close all; clc

prog_interval_sec = 2;   % 终端进度打印间隔 [s]
puma_prog('init', 'puma_sim', prog_interval_sec);

%% 1. 机器人模型
puma_prog('puma_sim', '加载 PUMA560 模型...', true);
mdl_puma560
p560 = p560.nofriction('all');   % rne 不含摩擦；摩擦由 Stribeck 单独计算
pnf = p560;
puma_prog('puma_sim', '模型就绪（无摩擦刚体 + Stribeck 配置）', true);

% Stribeck 摩擦参数（关节侧，索引 1..6 = joint1..6）
% 摩擦阻力矩（与速度反向）：tau_F = -sign(qd)*fc - (sign(qd)*fs-sign(qd)*fc)*lambda - fv*qd
% lambda_k = exp(-|qd_k/vs_k|^delta_k)；总力矩 tau = H*qdd + C*qd + g + tau_F
use_stribeck_friction = true;
stribeck_nominal = struct();
stribeck_nominal.fc = [1.20; 1.20; 1.00; 0.65; 0.45; 0.40];   % f_c^k [Nm] 库仑
stribeck_nominal.fs = [1.50; 1.50; 1.25; 0.85; 0.58; 0.52];   % f_s^k [Nm] 静摩擦（>fc）
stribeck_nominal.fv = [0.40; 0.32; 0.30; 0.26; 0.24; 0.20];   % f_v^k [Nm·s/rad] 粘滞
stribeck_nominal.vs = [0.05; 0.05; 0.05; 0.03; 0.03; 0.02];   % v_s^k [rad/s]
stribeck_nominal.delta = [1.2; 1.2; 1.2; 1.5; 1.5; 1.5];       % delta_s^k
stribeck_nominal.qd_eps = 1e-4;

% 按标签：轨迹中点 (t=T/2) 附近的随机偏置（台间差异，整条轨迹共享）
stribeck_rand = struct();
stribeck_rand.enable = true;
stribeck_rand.rng_seed = 20240523;
stribeck_rand.rel_std = struct( ...
    'fc', 0.02, 'fs', 0.02, 'fv', 0.04, 'vs', 0.02, 'delta', 0.015);
stribeck_rand.scale_min = 0.75;
stribeck_rand.scale_max = 1.25;
stribeck_rand.fs_over_fc_min = 1.05;

% 轨迹内缓慢漂移：t=0..T 上线性变化 + 可选低频振荡（模拟磨损/温漂）
stribeck_drift = struct();
stribeck_drift.enable = true;
stribeck_drift.rng_seed = stribeck_rand.rng_seed;
stribeck_drift.span = 0.12;          % 线性：u=0 -> 1-span, u=1 -> 1+span（再乘标签方向）
stribeck_drift.osc_hz = 0.012;       % 叠加正弦 [Hz]（T~10s 时不足 1 周期，变化平缓）
stribeck_drift.osc_amp = 0.035;      % 振荡相对幅值 ±3.5%
stribeck_drift.scale_min = 0.70;
stribeck_drift.scale_max = 1.30;
stribeck_drift.drift_vs_power = 0.5; % vs 漂移较缓：scale^0.5
stribeck_drift.drift_delta = false;  % delta 一般不漂移

if use_stribeck_friction
    fprintf('摩擦模型：Stribeck（标签初值 + 轨迹内缓慢漂移）\n');
    fprintf('  标称 @ t=T/2：\n');
    puma_print_stribeck_params(stribeck_nominal);
    if stribeck_drift.enable
        fprintf('  漂移：线性 span=%.0f%%', 100*stribeck_drift.span);
        if stribeck_drift.osc_amp > 0
            fprintf(' + 振荡 %.1f%% @ %.3f Hz', ...
                100*stribeck_drift.osc_amp, stribeck_drift.osc_hz);
        end
        fprintf('，尺度∈[%.2f,%.2f]\n', stribeck_drift.scale_min, stribeck_drift.scale_max);
    end
end

%% 2. 轨迹参数（一个字符 = 一条轨迹，采样点数 = n）
labels = {'e', 'v', 'q', 'b', 'o', 'l', 's', 'u', 'w', 'y', ...
    'a', 'z', 'h', 'm', 'p', 'c', 'd', 'r', 'g', 'n'};
% labels = {'e', 'v', 'q', 'b', 'o', 'l', 's', 'u'};
n = 2048;
dt = 0.005;
T_traj = (n - 1) * dt;
fourier_order = 5;
omega_f_hz = 0.25;              % 基频 [Hz]，文献典型取值（各标签相同）
omega_f = 2 * pi * omega_f_hz;

if exist('qz', 'var')
    q0 = reshape(qz(:), 6, 1);
else
    q0 = zeros(6, 1);
end

if isprop(p560, 'qlim') || isfield(p560, 'qlim')
    qlim = normalize_qlim(p560.qlim);
else
    qlim = [ ...
        -2.9671,  2.9671; ...
        -1.9199,  1.9199; ...
        -2.7925,  2.7925; ...
        -2.2176,  2.2176; ...
        -2.9671,  2.9671; ...
        -2.9671,  2.9671];
end
qd_max = 2.5;    % 关节速度上限 [rad/s]（优化约束）
qdd_max = 5.0;   % 关节加速度上限 [rad/s^2]（优化约束）

K = numel(labels);
traj = cell(1, K);
puma_prog('puma_sim', sprintf('开始生成 %d 条轨迹，每条 n=%d，T=%.2f s', K, n, T_traj), true);
fprintf('五阶傅里叶：f_1=%.4f Hz，f_max=%.4f Hz，n=%d\n', ...
    omega_f_hz, fourier_max_freq_hz(omega_f_hz, fourier_order), n);

for k = 1:K
    fprintf('  [%d/%d] label = ''%s''  n=%d\n', k, K, labels{k}, n);
    puma_prog('puma_sim', sprintf('轨迹 %d/%d 标签=''%s'' 傅里叶优化', k, K, labels{k}), true);
    [q, qd, qdd, fmeta] = joint_fourier_trajectory( ...
        labels{k}, n, dt, q0, omega_f, qlim, qd_max, qdd_max, fourier_order);
    puma_prog('puma_sim', sprintf('轨迹 %d/%d 标签=''%s'' 动力学 n=%d', k, K, labels{k}, n), true);
    traj{k} = puma_sim_dynamics(p560, pnf, q, qd, qdd, dt, ...
        stribeck_nominal, stribeck_rand, stribeck_drift, labels{k});
    traj{k}.fourier = fmeta;
    fprintf('      激励度 log(det(Cov))=%.2f，max|动力学分解误差|=%.3e [Nm]\n', ...
        fmeta.excitation_logdet, max(abs(traj{k}.dyn_err(:))));
end
fprintf('\n');
puma_prog('puma_sim', '导出 character_data.mat', true);

%% 3. character_data 导出
character_data = struct();
character_data.labels = labels(:);
character_data.dt = dt;
character_data.n = n;
character_data.n_dof = p560.n;
character_data.fourier_order = fourier_order;
character_data.omega_f_hz = omega_f_hz;
character_data.omega_f = omega_f;
character_data.T_traj = T_traj;
character_data.fourier_a = cell(K, 1);
character_data.fourier_b = cell(K, 1);
character_data.q_k0 = cell(K, 1);
character_data.t = cell(K, 1);
character_data.qp = cell(K, 1);
character_data.qv = cell(K, 1);
character_data.qa = cell(K, 1);
character_data.tau = cell(K, 1);
character_data.m = cell(K, 1);
character_data.c = cell(K, 1);
character_data.g = cell(K, 1);
character_data.p = cell(K, 1);
character_data.pdot = cell(K, 1);
character_data.mass_matrix = cell(K, 1);
character_data.friction = cell(K, 1);
character_data.friction_model = 'stribeck';
character_data.stribeck_nominal = stribeck_nominal;
character_data.stribeck_rand = stribeck_rand;
character_data.stribeck_drift = stribeck_drift;
character_data.stribeck = cell(K, 1);

for k = 1:K
    r = traj{k};
    character_data.t{k} = r.t(:);
    character_data.qp{k} = r.qp;
    character_data.qv{k} = r.qv;
    character_data.qa{k} = r.qa;
    character_data.tau{k} = r.tau;
    character_data.m{k} = r.m;
    character_data.c{k} = r.c;
    character_data.g{k} = r.g;
    character_data.p{k} = r.p;
    character_data.pdot{k} = r.pdot;
    character_data.mass_matrix{k} = r.mass_matrix;
    character_data.friction{k} = r.tau_F;
    character_data.stribeck{k} = r.stribeck;
    character_data.fourier_a{k} = r.fourier.a;
    character_data.fourier_b{k} = r.fourier.b;
    character_data.q_k0{k} = r.fourier.q_k0;
end

out_dir = fileparts(mfilename('fullpath'));
if isempty(out_dir)
    out_dir = pwd;
end
mat_file = fullfile(out_dir, 'character_data.mat');
save(mat_file, 'character_data', '-v7');
if exist(mat_file, 'file') ~= 2
    error('puma_sim:SaveFailed', '未能写入 %s', mat_file);
end
fprintf('已写出: %s\n', mat_file);

data_dir = fullfile(out_dir, 'data');
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end
fprintf('若需 pickle：python "%s"\n', fullfile(data_dir, 'mat_to_pickle.py'));

%% 4. 可视化
cols = lines(K);
tvec = (0:n-1) * dt;

figure('Name', '末端位置（fkine，非规划轨迹）', 'NumberTitle', 'off', 'Color', 'w');
hold on; grid on; axis equal
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
title('五阶傅里叶激励下的末端轨迹');
for k = 1:K
    Pk = traj{k}.P_ee;
    plot3(Pk(1, :), Pk(2, :), Pk(3, :), '-', 'Color', cols(k, :), 'LineWidth', 1.5, ...
        'DisplayName', labels{k});
end
legend('Location', 'bestoutside');
view([38 22]);

figure('Name', '关节角 q（rad）', 'NumberTitle', 'off', 'Color', 'w');
tl = tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for j = 1:6
    nexttile(tl);
    hold on; grid on
    for k = 1:K
        plot(tvec, traj{k}.qp(:, j), '-', 'Color', cols(k, :), 'LineWidth', 1.2, ...
            'DisplayName', labels{k});
    end
    ylabel(sprintf('q_%d', j));
    if j <= 2
        legend('Location', 'best');
    end
end
xlabel(tl, 't [s]');

fprintf('完成：K=%d，n=%d，dt=%.3f s，n_dof=%d。\n', K, n, dt, p560.n);
puma_prog('puma_sim', sprintf('全部完成 K=%d n=%d', K, n), true);

%% -------------------------------------------------------------------------
function puma_prog(channel, msg, force)
% 每 prog_interval 秒在终端打印一次进度；force=true 立即打印
% 初始化：puma_prog('init', 显示名, 间隔秒数)
    persistent S
    if nargin >= 1 && ischar(channel) && strcmp(channel, 'init')
        name = msg;
        iv = force;
        key = puma_prog_key(name);
        S.(key) = struct('name', name, 'iv', iv, 't0', tic, 'last', -iv);
        fprintf('[0.0s][%s] 程序启动（每 %.1f s 汇报进度）\n', name, iv);
        drawnow limitrate nocallbacks
        return
    end
    if nargin < 3
        force = false;
    end
    key = puma_prog_key(channel);
    if ~isfield(S, key)
        S.(key) = struct('name', channel, 'iv', 2, 't0', tic, 'last', -2);
    end
    st = S.(key);
    elapsed = toc(st.t0);
    if force || (elapsed - st.last >= st.iv)
        fprintf('[%.1fs][%s] %s\n', elapsed, st.name, msg);
        st.last = elapsed;
        S.(key) = st;
        drawnow limitrate nocallbacks
    end
end

function key = puma_prog_key(name)
    key = matlab.lang.makeValidName(['ch_' char(name)]);
end

%% -------------------------------------------------------------------------
function tau_f = stribeck_friction_joint(qd, prm)
% 式 (15)(16) Stribeck，输出为作用在关节上的阻力矩（与 qd 反向，qd>0 时 tau_F<0）
% tau_F = -sign(qd)*fc - (sign(qd)*fs - sign(qd)*fc)*lambda - fv*qd
    qd = qd(:);
    nj = numel(qd);
    tau_f = zeros(nj, 1);
    for k = 1:nj
        sk = stribeck_sign(qd(k), prm.qd_eps);
        lam = exp(-abs(qd(k) / prm.vs(k))^prm.delta(k));
        Fc = -sk * prm.fc(k);
        Fs = -sk * prm.fs(k);
        Fv = -qd(k) * prm.fv(k);
        tau_f(k) = Fc + (Fs - Fc) * lam + Fv;
    end
end

function s = stribeck_sign(qdk, eps_v)
    if abs(qdk) < eps_v
        s = 0;
    else
        s = sign(qdk);
    end
end

function puma_print_stribeck_params(prm)
    fprintf('  joint   fs[Nm]  fc[Nm]  fv[Nm·s/rad]  vs[rad/s]  delta\n');
    for k = 1:numel(prm.fc)
        fprintf('    %d    %5.2f   %5.2f      %5.3f       %5.3f    %4.1f\n', ...
            k, prm.fs(k), prm.fc(k), prm.fv(k), prm.vs(k), prm.delta(k));
    end
end

function base = stribeck_tag_base(nominal, rand_cfg, tag)
% 标签级基准（轨迹 t=T/2 时的目标水平），同 label 可复现
    base = nominal;
    if ~rand_cfg.enable
        return
    end
    rng_state = rng;
    seed = mod(rand_cfg.rng_seed + 9973 * fourier_label_index(tag) + tag_ascii_sum(tag), 2^31 - 1);
    rng(seed, 'twister');

    base.fc = stribeck_perturb_vec(nominal.fc, rand_cfg.rel_std.fc, rand_cfg);
    base.fs = stribeck_perturb_vec(nominal.fs, rand_cfg.rel_std.fs, rand_cfg);
    base.fv = stribeck_perturb_vec(nominal.fv, rand_cfg.rel_std.fv, rand_cfg);
    base.vs = stribeck_perturb_vec(nominal.vs, rand_cfg.rel_std.vs, rand_cfg);
    base.delta = stribeck_perturb_vec(nominal.delta, rand_cfg.rel_std.delta, rand_cfg);
    base.fs = max(base.fs, base.fc * rand_cfg.fs_over_fc_min);

    rng(rng_state);
end

function [dir_j, phase_j] = stribeck_drift_phases(tag, drift_cfg)
% 每条轨迹固定的漂移方向与振荡相位（6x1）
    nj = 6;
    rng_state = rng;
    seed = mod(drift_cfg.rng_seed + 17713 * fourier_label_index(tag) + tag_ascii_sum(tag), 2^31 - 1);
    rng(seed, 'twister');
    dir_j = sign(randn(nj, 1));
    dir_j(dir_j == 0) = 1;
    phase_j = 2 * pi * rand(nj, 1);
    rng(rng_state);
end

function scale = stribeck_drift_scale(t, T_sim, drift_cfg, dir_j, phase_j)
% 相对标签基准 (t=T/2) 的乘性漂移因子；线性 + 可选慢振荡
    if ~drift_cfg.enable
        scale = ones(size(dir_j));
        return
    end
    u = min(max(t / max(T_sim, eps), 0), 1);
    linear = 1 + drift_cfg.span * (u - 0.5) * 2 .* dir_j;
    scale = linear;
    if isfield(drift_cfg, 'osc_amp') && drift_cfg.osc_amp > 0
        osc = 1 + drift_cfg.osc_amp * sin(2 * pi * drift_cfg.osc_hz * t + phase_j);
        scale = scale .* osc;
    end
    scale = min(max(scale, drift_cfg.scale_min), drift_cfg.scale_max);
end

function prm = stribeck_params_at_time(base, drift_cfg, rand_cfg, t, T_sim, dir_j, phase_j)
% 将标签基准 base 按时刻 t 做缓慢漂移
    scale = stribeck_drift_scale(t, T_sim, drift_cfg, dir_j, phase_j);
    prm = base;
    prm.fc = base.fc .* scale;
    prm.fs = base.fs .* scale;
    prm.fv = base.fv .* scale;
    p_vs = drift_cfg.drift_vs_power;
    if isempty(p_vs) || p_vs <= 0
        prm.vs = base.vs;
    else
        prm.vs = base.vs .* (scale .^ p_vs);
    end
    if isfield(drift_cfg, 'drift_delta') && drift_cfg.drift_delta
        prm.delta = base.delta .* scale;
    else
        prm.delta = base.delta;
    end
    prm.fs = max(prm.fs, prm.fc * rand_cfg.fs_over_fc_min);
end

function x = stribeck_perturb_vec(x0, rel_std, rand_cfg)
% 对数正态乘性扰动，并限制在 [scale_min, scale_max] * 标称值
    x0 = x0(:);
    scale = exp(rel_std * randn(size(x0)));
    scale = min(max(scale, rand_cfg.scale_min), rand_cfg.scale_max);
    x = x0 .* scale;
end

function v = tag_ascii_sum(tag)
% 多字符标签（如 ''evqb''）转数值种子，避免 double(tag) 矩阵过大
    c = char(tag);
    if isempty(c)
        v = 0;
    else
        v = mod(sum(double(c)), 2^31 - 1);
    end
end

function idx = fourier_label_index(tag)
% 20 个字符标签 -> 1..20
    all_labels = {'e', 'v', 'q', 'b', 'o', 'l', 's', 'u', 'w', 'y', ...
        'a', 'z', 'h', 'm', 'p', 'c', 'd', 'r', 'g', 'n'};
    idx = find(strcmp(all_labels, tag), 1);
    if isempty(idx)
        idx = mod(tag_ascii_sum(tag), numel(all_labels)) + 1;
    end
end

function [a, b, q_off] = fourier_coeff_initial(tag, omega_f)
% 迭代优化初值：各 label 不同的五阶傅里叶系数与 q_{k,0} 偏置（相对 q0）
    idx = fourier_label_index(tag);
    nf = 5;
    a = zeros(6, nf);
    b = zeros(6, nf);
    amp_joint = [1.00; 1.12; 0.92; 1.08; 1.15; 0.96];
    amp_harm = 1 ./ (1:nf);
    for j = 1:6
        for i = 1:nf
            pa = 0.71 * idx + 1.27 * j + 0.53 * i + 0.19 * tag_ascii_sum(tag);
            pb = 1.03 * idx + 0.94 * j + 1.41 * i + 0.37 * tag_ascii_sum(tag) + pi / 5;
            base = (0.09 + 0.012 * idx) * amp_joint(j) * amp_harm(i);
            a(j, i) = base * sin(pa);
            b(j, i) = base * cos(pb);
        end
    end
    q_off = 0.08 * [ ...
        sin(0.38 * idx + 0.20); ...
        cos(0.42 * idx + 0.85); ...
        sin(0.51 * idx + 1.40); ...
        cos(0.46 * idx + 2.05); ...
        sin(0.55 * idx + 2.70); ...
        cos(0.49 * idx + 3.25)];
    % 初值缩放到可行域内
    iwf = (1:nf) * omega_f;
    for j = 1:6
        pos_exc = sum(abs(a(j, :)) ./ iwf + abs(b(j, :)) ./ iwf);
        vel_exc = sum(hypot(a(j, :), b(j, :)));
        s = min([1, 0.38 / max(pos_exc, 1e-9), 1.1 / max(vel_exc, 1e-9)]);
        a(j, :) = a(j, :) * s;
        b(j, :) = b(j, :) * s;
    end
end

function [q, qd, qdd] = eval_fourier_traj(a, b, q_k0, omega_f, n, dt)
% 由系数计算 q, qd, qdd；q_k0 为 6x1 的 q_{k,0}
    t = (0:n-1) * dt;
    nf = size(a, 2);
    q = zeros(6, n);
    qd = zeros(6, n);
    qdd = zeros(6, n);
    for j = 1:6
        q(j, :) = q_k0(j);
        for i = 1:nf
            wt = i * omega_f * t;
            denom = i * omega_f;
            q(j, :) = q(j, :) + (a(j, i) / denom) * sin(wt) - (b(j, i) / denom) * cos(wt);
            qd(j, :) = qd(j, :) + a(j, i) * cos(wt) + b(j, i) * sin(wt);
            qdd(j, :) = qdd(j, :) + i * omega_f * (-a(j, i) * sin(wt) + b(j, i) * cos(wt));
        end
    end
end

function ok = fourier_traj_feasible(q, qd, qdd, qlim, qd_max, qdd_max)
    qlim = normalize_qlim(qlim);
    ok = all(all(q >= qlim(:, 1) & q <= qlim(:, 2)));
    ok = ok && all(abs(qd(:)) <= qd_max) && all(abs(qdd(:)) <= qdd_max);
end

function qlim = normalize_qlim(qlim)
% 统一为 6x2，[q_min, q_max] 每行一个关节
    if size(qlim, 1) == 2 && size(qlim, 2) == 6
        qlim = qlim';
    end
end

function J = fourier_excitation_score(q, qd, qdd)
% 激励度：关节位置/速度/加速度样本协方差阵 log(det)，越大越利于辨识
    X = [q', qd', qdd'];
    C = (X - mean(X, 1))' * (X - mean(X, 1)) / max(size(X, 1) - 1, 1);
    C = C + 1e-8 * eye(size(C, 1));
    J = log(max(det(C), 1e-30));
end

function [a, b, q_k0, Jbest] = iterative_optimize_fourier(tag, q0, omega_f, n, dt, qlim, qd_max, qdd_max)
% 迭代优化 a_{k,i}, b_{k,i} 与 q_{k,0}（在 qlim 与速度/加速度约束内最大化激励度）
    idx = fourier_label_index(tag);
    rng_state = rng;
    rng(mod(idx * 104729 + tag_ascii_sum(tag), 2^31 - 1), 'twister');

    [a, b, q_off] = fourier_coeff_initial(tag, omega_f);
    q_k0 = q0(:) + q_off;
    q_k0 = min(max(q_k0, qlim(:, 1) + 0.05), qlim(:, 2) - 0.05);

    [q, qd, qdd] = eval_fourier_traj(a, b, q_k0, omega_f, n, dt);
    Jbest = -inf;
    if fourier_traj_feasible(q, qd, qdd, qlim, qd_max, qdd_max)
        Jbest = fourier_excitation_score(q, qd, qdd);
    end
    best_a = a;
    best_b = b;
    best_q = q_k0;

    max_iter = 120;
    step0 = 0.10;
    nf = 5;
    puma_prog('puma_sim', sprintf('傅里叶优化 label=''%s'' n=%d', tag, n), true);
    for iter = 1:max_iter
        puma_prog('puma_sim', sprintf('傅里叶优化 ''%s'' 迭代 %d/%d', tag, iter, max_iter));
        step = step0 * (1 - 0.88 * (iter - 1) / max_iter);
        da = step * randn(6, nf) * (0.08 + 0.01 * idx);
        db = step * randn(6, nf) * (0.08 + 0.01 * idx);
        dq = step * 0.04 * randn(6, 1);

        a_try = a + da;
        b_try = b + db;
        q_try = min(max(q_k0 + dq, qlim(:, 1) + 0.02), qlim(:, 2) - 0.02);

        [q, qd, qdd] = eval_fourier_traj(a_try, b_try, q_try, omega_f, n, dt);
        if ~fourier_traj_feasible(q, qd, qdd, qlim, qd_max, qdd_max)
            continue
        end
        J = fourier_excitation_score(q, qd, qdd);
        if J > Jbest
            Jbest = J;
            best_a = a_try;
            best_b = b_try;
            best_q = q_try;
            a = a_try;
            b = b_try;
            q_k0 = q_try;
        end
    end

    a = best_a;
    b = best_b;
    q_k0 = best_q;
    rng(rng_state);
end

function f_hz = fourier_max_freq_hz(omega_f_hz, nf)
% 五阶（nf 阶）傅里叶轨迹的理论最高频率：f_max = nf * f_1 [Hz]
    f_hz = nf * omega_f_hz(:);
end

function puma_print_segment_freq(seg_labels, omega_f_hz_list, nf, title_str, n_pts, dt, n_cycles, omega_f_mode)
% 终端打印每段基频 f_1、最高谐波 f_max（= nf×f_1）；可选显示 n_seg、T_seg、周期数
    if nargin < 4 || isempty(title_str)
        title_str = '';
    end
    seg_labels = cellstr(seg_labels(:));
    f1 = omega_f_hz_list(:);
    ns = numel(seg_labels);
    if numel(f1) ~= ns
        error('puma_print_segment_freq:Size', '标签数与 omega_f_hz 长度不一致');
    end
    fmax = fourier_max_freq_hz(f1, nf);
    show_seg = nargin >= 6 && ~isempty(n_pts) && ~isempty(dt);
    if show_seg
        T_one = (n_pts - 1) * dt;
        f_nyq = 1 / (2 * dt);
    end
    if ~isempty(title_str)
        fprintf('【各段最高频率】%s（f_max = %d × f_1，与采样点数无关除非用 from_cycles）\n', title_str, nf);
    else
        fprintf('【各段最高频率】f_max = %d × f_1\n', nf);
    end
    if show_seg
        fprintf('  n_seg=%d  T_seg=%.3f s  dt=%.4f s  f_Nyquist=%.2f Hz', n_pts, T_one, dt, f_nyq);
        if nargin >= 8 && ~isempty(omega_f_mode)
            fprintf('  omega_f_mode=%s', omega_f_mode);
        end
        fprintf('\n');
    end
    if nargin >= 7 && ~isempty(n_cycles) && numel(n_cycles) >= ns
        fprintf('  段号  label    周期数    f_1 [Hz]    f_max [Hz]\n');
        for s = 1:ns
            fprintf('  %2d    %-6s   %6.1f     %8.4f     %8.4f\n', ...
                s, seg_labels{s}, n_cycles(s), f1(s), fmax(s));
        end
    else
        fprintf('  段号  label    f_1 [Hz]    f_max [Hz]\n');
        for s = 1:ns
            fprintf('  %2d    %-6s   %8.4f     %8.4f\n', s, seg_labels{s}, f1(s), fmax(s));
        end
    end
end

function [q, qd, qdd, meta] = joint_fourier_trajectory(tag, n, dt, q0, omega_f, qlim, qd_max, qdd_max, nf)
% 式 (29)；系数由 iterative_optimize_fourier 确定
    if nargin < 10 || isempty(nf)
        nf = 5;
    end
    [a, b, q_k0, Jopt] = iterative_optimize_fourier(tag, q0, omega_f, n, dt, qlim, qd_max, qdd_max);
    [q, qd, qdd] = eval_fourier_traj(a, b, q_k0, omega_f, n, dt);
    omega_f_hz = omega_f / (2 * pi);
    meta = struct('a', a, 'b', b, 'q_k0', q_k0, 'omega_f', omega_f, ...
        'omega_f_hz', omega_f_hz, 'fourier_order', nf, ...
        'f_max_hz', fourier_max_freq_hz(omega_f_hz, nf), 'excitation_logdet', Jopt);
end

function out = puma_sim_dynamics(p560, pnf, q, qd, qdd, dt, nominal, rand_cfg, drift_cfg, tag)
    n = size(q, 2);
    t = (0:n-1) * dt;
    T_sim = (n - 1) * dt;
    base = stribeck_tag_base(nominal, rand_cfg, tag);
    [dir_j, phase_j] = stribeck_drift_phases(tag, drift_cfg);

    tau = zeros(6, n);
    tau_H = zeros(6, n);
    tau_C = zeros(6, n);
    tau_G = zeros(6, n);
    tau_F = zeros(6, n);
    mass_matrix = zeros(n, 6, 6);
    p_joint = zeros(n, 6);
    fc_hist = zeros(n, 6);
    scale_hist = zeros(n, 6);
    puma_prog('puma_sim', sprintf('动力学 tag=''%s'' n=%d (rne+fkine)', tag, n), true);
    for k = 1:n
        puma_prog('puma_sim', sprintf('动力学 ''%s'' 采样 %d/%d (%.1f%%)', tag, k, n, 100*k/n));
        qr = q(:, k)';
        qdr = qd(:, k)';
        qddr = qdd(:, k)';
        prm_t = stribeck_params_at_time(base, drift_cfg, rand_cfg, t(k), T_sim, dir_j, phase_j);
        fc_hist(k, :) = prm_t.fc';
        scale_hist(k, :) = (prm_t.fc ./ max(base.fc, 1e-9))';
        tau_H(:, k) = pnf.itorque(qr, qddr)';
        tau_G(:, k) = pnf.gravload(qr)';
        Ck = pnf.coriolis(qr, qdr);
        tau_C(:, k) = (Ck * qdr')';
        tau_F(:, k) = stribeck_friction_joint(qd(:, k), prm_t);
        tau(:, k) = tau_H(:, k) + tau_C(:, k) + tau_G(:, k) + tau_F(:, k);
        Mk = pnf.inertia(qr);
        mass_matrix(k, :, :) = Mk;
        p_joint(k, :) = (Mk * qdr')';
    end
    pdot_joint = zeros(n, 6);
    for k = 2:n-1
        pdot_joint(k, :) = (p_joint(k+1, :) - p_joint(k-1, :)) / (2*dt);
    end
    pdot_joint(1, :) = (p_joint(2, :) - p_joint(1, :)) / dt;
    pdot_joint(n, :) = (p_joint(n, :) - p_joint(n-1, :)) / dt;

    dyn_err = tau - (tau_H + tau_C + tau_G + tau_F);

    P_ee = zeros(3, n);
    for k = 1:n
        puma_prog('puma_sim', sprintf('末端 fkine ''%s'' %d/%d', tag, k, n));
        Tk = p560.fkine(q(:, k)');
        if isa(Tk, 'SE3')
            P_ee(:, k) = Tk.t;
        else
            P_ee(:, k) = transl(Tk)';
        end
    end

    out = struct();
    out.t = t(:);
    out.qp = q';
    out.qv = qd';
    out.qa = qdd';
    out.tau = tau';
    out.m = tau_H';
    out.c = tau_C';
    out.g = tau_G';
    out.tau_F = tau_F';
    out.stribeck = struct('base', base, 'fc_hist', fc_hist, 'scale_hist', scale_hist, 'tag', tag);
    out.p = p_joint;
    out.pdot = pdot_joint;
    out.mass_matrix = mass_matrix;
    out.dyn_err = dyn_err;
    out.P_ee = P_ee;
end