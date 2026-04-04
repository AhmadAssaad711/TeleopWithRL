function out = run_matlab_open_loop_plant_io(varargin)
%RUN_MATLAB_OPEN_LOOP_PLANT_IO Run the normal MATLAB plant model open loop.
%
% Builds a temporary harness around the Plant subsystem in the reference
% Simulink model, drives it with:
%   F_h(t) = force_amp * sin(2*pi*force_freq*t + force_phase)
%   u(t)   = u_voltage
% and exports:
%   - matlab_open_loop_io.csv
%   - matlab_open_loop_debug.mat
%   - matlab_open_loop_scopes.png
%   - matlab_open_loop_error_scopes.png
%   - matlab_open_loop_summary.txt
%
% The source model is never saved or modified.

    repo_root = fileparts(fileparts(mfilename('fullpath')));
    reference_dir = fullfile(fileparts(mfilename('fullpath')), 'reference');

    parser = inputParser;
    parser.addParameter('mdl_path', fullfile(reference_dir, 'Simulation_NonlinearModel_PIRL_final.slx'));
    parser.addParameter('init_script', fullfile(reference_dir, 'SI_NonLinear.m'));
    parser.addParameter('out_dir', fullfile(repo_root, 'results_fh', 'matlab_open_loop_plant_io'));
    parser.addParameter('duration', 180.0);
    parser.addParameter('switch_time', 30.0);
    parser.addParameter('force_amp', 10.0);
    parser.addParameter('force_freq', 0.5);
    parser.addParameter('force_phase', 0.0);
    parser.addParameter('u_voltage', 0.0);
    parser.addParameter('Ts', 0.0005);
    parser.addParameter('override_script_ts', false);
    parser.parse(varargin{:});
    opts = parser.Results;

    mdl_path = char(opts.mdl_path);
    if ~isfile(mdl_path)
        error('Source model not found: %s', mdl_path);
    end

    out_dir = char(opts.out_dir);
    if ~isfolder(out_dir)
        mkdir(out_dir);
    end

    [~, source_model, ~] = fileparts(mdl_path);
    harness = 'matlab_open_loop_plant_harness_autogen';
    cleanup_obj = onCleanup(@() cleanup_models(source_model, harness)); %#ok<NASGU>

    init_script = char(opts.init_script);
    if ~isempty(init_script)
        run_init_script(init_script, mdl_path);
        params = snapshot_parameters_from_base();
        if opts.override_script_ts
            params.Ts = opts.Ts;
        end
    else
        params = default_plant_parameters(opts.Ts);
    end
    t = (0:params.Ts:opts.duration).';
    fh = opts.force_amp * sin((2.0 * pi * opts.force_freq * t) + opts.force_phase);
    u = opts.u_voltage * ones(size(t));

    assignin('base', 'fh_input_ts', timeseries(fh, t));
    assignin('base', 'u_input_ts', timeseries(u, t));
    param_names = fieldnames(params);
    for idx = 1:numel(param_names)
        assignin('base', param_names{idx}, params.(param_names{idx}));
    end

    load_system(mdl_path);
    load_system('simulink');
    if bdIsLoaded(harness)
        close_system(harness, 0);
    end

    new_system(harness);
    add_block('simulink/Sources/From Workspace', [harness '/F_h_in'], ...
        'VariableName', 'fh_input_ts', ...
        'Position', [40 70 155 100]);
    add_block('simulink/Sources/From Workspace', [harness '/u_in'], ...
        'VariableName', 'u_input_ts', ...
        'Position', [40 150 155 180]);
    add_block([source_model '/Plant'], [harness '/Plant'], ...
        'Position', [240 40 470 300]);

    plant_info = configure_plant_outputs([harness '/Plant']);

    add_to_workspace_block(harness, 'F_h_ws', 'F_h', [560 30 700 60]);
    add_to_workspace_block(harness, 'u_ws', 'u', [560 70 700 100]);
    add_to_workspace_block(harness, 'x_s_ws', 'x_s', [560 110 700 140]);
    add_to_workspace_block(harness, 'x_m_ws', 'x_m', [560 150 700 180]);
    add_to_workspace_block(harness, 'x_sdot_ws', 'x_sdot', [560 190 700 220]);
    add_to_workspace_block(harness, 'x_mdot_ws', 'x_mdot', [560 230 700 260]);
    add_to_workspace_block(harness, 'K_e_ws', 'K_e_sig', [560 310 700 340]);
    add_to_workspace_block(harness, 'B_e_ws', 'B_e_sig', [560 350 700 380]);
    if plant_info.has_direct_Fe
        add_to_workspace_block(harness, 'Fe_ws', 'Fe', [560 270 700 300]);
    end

    add_line(harness, 'F_h_in/1', 'Plant/1', 'autorouting', 'on');
    add_line(harness, 'u_in/1', 'Plant/2', 'autorouting', 'on');
    add_line(harness, 'F_h_in/1', 'F_h_ws/1', 'autorouting', 'on');
    add_line(harness, 'u_in/1', 'u_ws/1', 'autorouting', 'on');
    add_line(harness, sprintf('Plant/%d', plant_info.x_s_port), 'x_s_ws/1', 'autorouting', 'on');
    add_line(harness, sprintf('Plant/%d', plant_info.x_m_port), 'x_m_ws/1', 'autorouting', 'on');
    add_line(harness, sprintf('Plant/%d', plant_info.x_sdot_port), 'x_sdot_ws/1', 'autorouting', 'on');
    add_line(harness, sprintf('Plant/%d', plant_info.x_mdot_port), 'x_mdot_ws/1', 'autorouting', 'on');
    add_line(harness, sprintf('Plant/%d', plant_info.K_e_port), 'K_e_ws/1', 'autorouting', 'on');
    add_line(harness, sprintf('Plant/%d', plant_info.B_e_port), 'B_e_ws/1', 'autorouting', 'on');
    if plant_info.has_direct_Fe
        add_line(harness, sprintf('Plant/%d', plant_info.Fe_port), 'Fe_ws/1', 'autorouting', 'on');
    end

    set_param(harness, ...
        'StartTime', '0.0', ...
        'StopTime', num2str(opts.duration, '%.16g'), ...
        'Solver', 'ode4', ...
        'FixedStep', 'Ts', ...
        'SaveOutput', 'off', ...
        'SaveTime', 'off', ...
        'ReturnWorkspaceOutputs', 'off');

    sim(harness);

    F_h_ts = F_h;
    u_ts = u;
    x_s_ts = x_s;
    x_m_ts = x_m;
    x_sdot_ts = x_sdot;
    x_mdot_ts = x_mdot;
    K_e_ts = K_e_sig;
    B_e_ts = B_e_sig;
    if plant_info.has_direct_Fe
        Fe_ts = Fe;
    else
        Fe_ts = [];
    end

    data = collect_output_data(F_h_ts, u_ts, x_s_ts, x_m_ts, x_sdot_ts, x_mdot_ts, K_e_ts, B_e_ts, opts.switch_time, Fe_ts);
    results_full = build_results_table(data);
    results_20ms = results_full(1:round(0.02 / opts.Ts):height(results_full), :);

    metadata = struct( ...
        'source_model', mdl_path, ...
        'init_script', init_script, ...
        'harness_model', harness, ...
        'duration_s', opts.duration, ...
        'sample_time_s', params.Ts, ...
        'switch_time_s', opts.switch_time, ...
        'solver', 'ode4', ...
        'fe_source', ternary_string(plant_info.has_direct_Fe, 'direct_model_output', 'reconstructed_from_xs_and_xsdot'), ...
        'force_amp_n', opts.force_amp, ...
        'force_freq_hz', opts.force_freq, ...
        'force_phase_rad', opts.force_phase, ...
        'u_voltage_v', opts.u_voltage, ...
        'parameter_source', ternary_string(isempty(init_script), 'hardcoded_defaults', 'matlab_init_script'));

    csv_path = fullfile(out_dir, 'matlab_open_loop_io.csv');
    mat_path = fullfile(out_dir, 'matlab_open_loop_debug.mat');
    scopes_path = fullfile(out_dir, 'matlab_open_loop_scopes.png');
    error_scopes_path = fullfile(out_dir, 'matlab_open_loop_error_scopes.png');
    summary_path = fullfile(out_dir, 'matlab_open_loop_summary.txt');

    writetable(results_full, csv_path);
    save(mat_path, 'results_full', 'results_20ms', 'metadata', 'params', 'data');
    plot_main_scopes(scopes_path, data, opts.switch_time);
    plot_error_scopes(error_scopes_path, data, opts.switch_time);
    write_summary(summary_path, data, metadata);

    fprintf('MATLAB open-loop plant I/O run complete.\n');
    fprintf('Samples (full): %d\n', height(results_full));
    fprintf('Samples (20 ms): %d\n', height(results_20ms));
    fprintf('CSV: %s\n', csv_path);
    fprintf('MAT: %s\n', mat_path);
    fprintf('Scopes: %s\n', scopes_path);
    fprintf('Error scopes: %s\n', error_scopes_path);
    fprintf('Summary: %s\n', summary_path);

    out = struct( ...
        'csv_path', csv_path, ...
        'mat_path', mat_path, ...
        'scopes_path', scopes_path, ...
        'error_scopes_path', error_scopes_path, ...
        'summary_path', summary_path, ...
        'samples_full', height(results_full), ...
        'samples_20ms', height(results_20ms));
end


function params = default_plant_parameters(Ts)
    params = struct( ...
        'Ts', Ts, ...
        'A_p', 4.2072e-4, ...
        'A_t', 1.257e-5, ...
        'b_v', 0.21, ...
        'C_v', 4.5e-9, ...
        'D_t', 4e-3, ...
        'L_t', 10.0, ...
        'l_cyl', 0.275, ...
        'm_p', 0.25, ...
        'V_md', 2e-6, ...
        'V_sd', 2e-6, ...
        'beta', 11.6, ...
        'mui', 1.813e-5, ...
        'R', 287.0, ...
        'rho0', 1.204, ...
        'T', 293.0, ...
        'P_atm', 101325.0, ...
        'P_s', 600000.0, ...
        'omega_v', 100.0, ...
        'zeta_v', 0.7, ...
        'K_v', 0.1, ...
        'K_e', 331.0, ...
        'B_e', 0.003, ...
        'P_md', 101325.0, ...
        'k_h', 0.0, ...
        'B_h', 0.0);
end


function run_init_script(init_script, mdl_path)
    [script_folder, script_name, script_ext] = fileparts(init_script);
    if isempty(script_ext)
        script_path = [init_script '.m'];
    else
        script_path = init_script;
    end
    if ~isfile(script_path)
        mdl_folder = fileparts(mdl_path);
        alt_path = fullfile(mdl_folder, script_path);
        if isfile(alt_path)
            script_path = alt_path;
            [script_folder, script_name, ~] = fileparts(script_path);
        else
            error('Initialization script not found: %s', init_script);
        end
    end

    current_dir = pwd;
    cleaner = onCleanup(@() cd(current_dir)); %#ok<NASGU>
    if ~isempty(script_folder)
        cd(script_folder);
    end
    evalin('base', sprintf('run(''%s'');', strrep(script_path, '\', '/')));
end


function params = snapshot_parameters_from_base()
    names = { ...
        'Ts', 'A_p', 'A_t', 'b_v', 'C_v', 'D_t', 'L_t', 'l_cyl', 'm_p', ...
        'V_md', 'V_sd', 'beta', 'mui', 'R', 'rho0', 'T', 'P_atm', 'P_s', ...
        'omega_v', 'zeta_v', 'K_v', 'K_e', 'B_e', 'P_md', 'k_h', 'B_h'};
    params = struct();
    for idx = 1:numel(names)
        name = names{idx};
        params.(name) = evalin('base', name);
    end
end


function info = configure_plant_outputs(plant_path)
    outports = find_system(plant_path, 'SearchDepth', 1, 'BlockType', 'Outport');
    existing_names = cell(size(outports));
    existing_ports = zeros(size(outports));
    for idx = 1:numel(outports)
        existing_names{idx} = char(get_param(outports{idx}, 'Name'));
        existing_ports(idx) = str2double(get_param(outports{idx}, 'Port'));
    end

    info = struct();
    info.x_s_port = lookup_port(existing_names, existing_ports, 'x_s');
    info.x_m_port = lookup_port(existing_names, existing_ports, 'x_m');
    info.has_direct_Fe = has_named_port(existing_names, 'Fe');
    info.Fe_port = lookup_port(existing_names, existing_ports, 'Fe');

    next_port = max([existing_ports(:); 0]) + 1;
    if ~has_named_port(existing_names, 'x_sdot')
        add_block('simulink/Sinks/Out1', [plant_path '/x_sdot'], ...
            'Port', num2str(next_port), ...
            'Position', [760 120 790 134]);
        add_line(plant_path, 'Slave cylinder/1', 'x_sdot/1', 'autorouting', 'on');
        existing_names{end + 1} = 'x_sdot'; %#ok<AGROW>
        existing_ports(end + 1) = next_port; %#ok<AGROW>
        next_port = next_port + 1;
    end

    if ~has_named_port(existing_names, 'x_mdot')
        add_block('simulink/Sinks/Out1', [plant_path '/x_mdot'], ...
            'Port', num2str(next_port), ...
            'Position', [760 160 790 174]);
        add_line(plant_path, 'Master cylinder/2', 'x_mdot/1', 'autorouting', 'on');
        existing_names{end + 1} = 'x_mdot'; %#ok<AGROW>
        existing_ports(end + 1) = next_port; %#ok<AGROW>
        next_port = next_port + 1;
    end

    if ~has_named_port(existing_names, 'K_e_sig')
        add_block('simulink/Sinks/Out1', [plant_path '/K_e_sig'], ...
            'Port', num2str(next_port), ...
            'Position', [760 200 790 214]);
        add_line(plant_path, 'Sum4/1', 'K_e_sig/1', 'autorouting', 'on');
        existing_names{end + 1} = 'K_e_sig'; %#ok<AGROW>
        existing_ports(end + 1) = next_port; %#ok<AGROW>
        next_port = next_port + 1;
    end

    if ~has_named_port(existing_names, 'B_e_sig')
        add_block('simulink/Sinks/Out1', [plant_path '/B_e_sig'], ...
            'Port', num2str(next_port), ...
            'Position', [760 240 790 254]);
        add_line(plant_path, 'Sum5/1', 'B_e_sig/1', 'autorouting', 'on');
        existing_names{end + 1} = 'B_e_sig'; %#ok<AGROW>
        existing_ports(end + 1) = next_port; %#ok<AGROW>
    end

    info.x_sdot_port = lookup_port(existing_names, existing_ports, 'x_sdot');
    info.x_mdot_port = lookup_port(existing_names, existing_ports, 'x_mdot');
    info.K_e_port = lookup_port(existing_names, existing_ports, 'K_e_sig');
    info.B_e_port = lookup_port(existing_names, existing_ports, 'B_e_sig');
end


function add_to_workspace_block(model, block_name, variable_name, position)
    add_block('simulink/Sinks/To Workspace', [model '/' block_name], ...
        'VariableName', variable_name, ...
        'SaveFormat', 'Timeseries', ...
        'MaxDataPoints', 'inf', ...
        'Position', position);
end


function data = collect_output_data(F_h_ts, u_ts, x_s_ts, x_m_ts, x_sdot_ts, x_mdot_ts, K_e_ts, B_e_ts, switch_time, Fe_ts)
    data = struct();
    data.t = to_column(x_m_ts.Time);
    data.F_h = to_column(F_h_ts.Data);
    data.u = to_column(u_ts.Data);
    data.x_s = to_column(x_s_ts.Data);
    data.x_m = to_column(x_m_ts.Data);
    data.x_sdot = to_column(x_sdot_ts.Data);
    data.x_mdot = to_column(x_mdot_ts.Data);
    data.K_e = to_column(K_e_ts.Data);
    data.B_e = to_column(B_e_ts.Data);
    if nargin >= 10 && ~isempty(Fe_ts)
        data.F_e = to_column(Fe_ts.Data);
    else
        data.F_e = (data.K_e .* data.x_s) + (data.B_e .* data.x_sdot);
    end
    data.e = data.x_m - data.x_s;
    data.transparency_error = (data.F_e .* data.x_mdot) - (data.F_h .* data.x_sdot);
    data.env_label = strings(size(data.t));
    data.env_label(:) = "skin";
    data.env_label(data.t >= switch_time) = "fat";
end


function results = build_results_table(data)
    results = table( ...
        data.t, ...
        data.F_h, ...
        data.u, ...
        data.x_m, ...
        data.x_s, ...
        data.x_mdot, ...
        data.x_sdot, ...
        data.F_e, ...
        data.e, ...
        data.transparency_error, ...
        'VariableNames', { ...
            't', 'F_h', 'u', 'x_m', 'x_s', 'x_mdot', 'x_sdot', 'F_e', 'e', 'transparency_error'});
end


function plot_main_scopes(out_path, data, switch_time)
    fig = figure('Visible', 'off', 'Position', [100 100 1400 1200]);
    tl = tiledlayout(7, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    plot_signal(nexttile(tl), data.t, data.F_h, switch_time, 'Input force [N]');
    plot_signal(nexttile(tl), data.t, data.u, switch_time, 'Valve input [V]');
    plot_signal(nexttile(tl), data.t, data.x_m, switch_time, 'Master position [m]');
    plot_signal(nexttile(tl), data.t, data.x_s, switch_time, 'Slave position [m]');
    plot_signal(nexttile(tl), data.t, data.x_mdot, switch_time, 'Master velocity [m/s]');
    plot_signal(nexttile(tl), data.t, data.x_sdot, switch_time, 'Slave velocity [m/s]');
    plot_signal(nexttile(tl), data.t, data.F_e, switch_time, 'Environment force [N]');

    title(tl, 'MATLAB Open-Loop Plant I/O Scopes');
    xlabel(tl, 'Time [s]');
    exportgraphics(fig, out_path, 'Resolution', 160);
    close(fig);
end


function plot_error_scopes(out_path, data, switch_time)
    fig = figure('Visible', 'off', 'Position', [100 100 1400 650]);
    tl = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax1 = nexttile(tl);
    plot(ax1, data.t, data.e, 'LineWidth', 1.3, 'Color', [0.85 0.2 0.2]);
    yline(ax1, 0.0, 'Color', [0.1 0.1 0.1], 'LineWidth', 0.8);
    xline(ax1, switch_time, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.0);
    ylabel(ax1, 'Tracking error [m]');
    grid(ax1, 'on');

    ax2 = nexttile(tl);
    plot(ax2, data.t, data.transparency_error, 'LineWidth', 1.3, 'Color', [0.2 0.35 0.85]);
    yline(ax2, 0.0, 'Color', [0.1 0.1 0.1], 'LineWidth', 0.8);
    xline(ax2, switch_time, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.0);
    ylabel(ax2, 'Transparency error [W]');
    grid(ax2, 'on');

    title(tl, 'MATLAB Tracking And Transparency Error Scopes');
    xlabel(tl, 'Time [s]');
    exportgraphics(fig, out_path, 'Resolution', 160);
    close(fig);
end


function write_summary(out_path, data, metadata)
    fid = fopen(out_path, 'w');
    if fid == -1
        error('Failed to open summary file: %s', out_path);
    end
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid, 'MATLAB open-loop plant I/O analysis\n');
    fprintf(fid, '==================================\n');
    fprintf(fid, 'source_model=%s\n', metadata.source_model);
    fprintf(fid, 'init_script=%s\n', metadata.init_script);
    fprintf(fid, 'parameter_source=%s\n', metadata.parameter_source);
    fprintf(fid, 'duration_s=%.6f\n', metadata.duration_s);
    fprintf(fid, 'sample_time_s=%.6f\n', metadata.sample_time_s);
    fprintf(fid, 'switch_time_s=%.6f\n', metadata.switch_time_s);
    fprintf(fid, 'solver=%s\n', metadata.solver);
    fprintf(fid, 'fe_source=%s\n', metadata.fe_source);
    fprintf(fid, 'force_amp_n=%.6f\n', metadata.force_amp_n);
    fprintf(fid, 'force_freq_hz=%.6f\n', metadata.force_freq_hz);
    fprintf(fid, 'force_phase_rad=%.6f\n', metadata.force_phase_rad);
    fprintf(fid, 'u_voltage_v=%.6f\n\n', metadata.u_voltage_v);

    segments = {
        'full_run', true(size(data.t));
        'skin_segment', data.t < metadata.switch_time_s;
        'fat_segment', data.t >= metadata.switch_time_s
    };
    signal_names = {'F_h', 'u', 'x_m', 'x_s', 'x_mdot', 'x_sdot', 'F_e', 'e', 'transparency_error', 'K_e', 'B_e'};

    for seg_idx = 1:size(segments, 1)
        seg_name = segments{seg_idx, 1};
        mask = segments{seg_idx, 2};
        fprintf(fid, '[%s]\n', seg_name);
        fprintf(fid, 'samples=%d\n', nnz(mask));
        if ~any(mask)
            fprintf(fid, 'empty=true\n\n');
            continue;
        end
        fprintf(fid, 'time_start_s=%.6f\n', data.t(find(mask, 1, 'first')));
        fprintf(fid, 'time_end_s=%.6f\n', data.t(find(mask, 1, 'last')));
        for sig_idx = 1:numel(signal_names)
            sig_name = signal_names{sig_idx};
            values = data.(sig_name)(mask);
            fprintf(fid, ...
                '%s: min=%.9f, max=%.9f, mean=%.9f, rms=%.9f, peak_to_peak=%.9f\n', ...
                sig_name, ...
                min(values), ...
                max(values), ...
                mean(values), ...
                sqrt(mean(values .^ 2)), ...
                max(values) - min(values));
        end
        fprintf(fid, '\n');
    end
end


function tf = has_named_port(names, target_name)
    tf = any(strcmp(names, target_name));
end


function port = lookup_port(names, ports, target_name)
    idx = find(strcmp(names, target_name), 1, 'first');
    if isempty(idx)
        port = [];
    else
        port = ports(idx);
    end
end


function value = ternary_string(condition, true_value, false_value)
    if condition
        value = true_value;
    else
        value = false_value;
    end
end


function plot_signal(ax, t, y, switch_time, ylabel_text)
    plot(ax, t, y, 'LineWidth', 1.2, 'Color', [0.1 0.4 0.8]);
    xline(ax, switch_time, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.0);
    ylabel(ax, ylabel_text);
    grid(ax, 'on');
end


function y = to_column(x)
    y = squeeze(x);
    if isrow(y)
        y = y.';
    end
end


function cleanup_models(source_model, harness)
    if bdIsLoaded(harness)
        close_system(harness, 0);
    end
    if bdIsLoaded(source_model)
        close_system(source_model, 0);
    end
end
