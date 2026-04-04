function out = run_matlab_top_level_as_run(varargin)
%RUN_MATLAB_TOP_LEVEL_AS_RUN Run the top-level Simulink model as configured.
%
% Loads the requested model and init script, preserves the model's own
% top-level source path and solver settings, forces the control input block
% to a constant zero if requested, taps the live input/output signals, and
% exports:
%   - matlab_top_level_io.csv
%   - matlab_top_level_debug.mat
%   - matlab_top_level_scopes.png
%   - matlab_top_level_error_scopes.png
%   - matlab_top_level_summary.txt
%
% The source model is never saved or modified on disk.

    repo_root = fileparts(fileparts(mfilename('fullpath')));
    reference_dir = fullfile(fileparts(mfilename('fullpath')), 'reference');

    parser = inputParser;
    parser.addParameter('mdl_path', fullfile(reference_dir, 'Simulation_NonlinearModel_PIRL_final.slx'));
    parser.addParameter('init_script', fullfile(reference_dir, 'SI_NonLinear.m'));
    parser.addParameter('out_dir', fullfile(repo_root, 'results_fh', 'matlab_top_level_as_run'));
    parser.addParameter('switch_time', 30.0);
    parser.addParameter('u_voltage', 0.0);
    parser.addParameter('force_control_zero', true);
    parser.parse(varargin{:});
    opts = parser.Results;

    mdl_path = char(opts.mdl_path);
    if ~isfile(mdl_path)
        error('Source model not found: %s', mdl_path);
    end

    init_script = char(opts.init_script);
    if ~isempty(init_script) && ~isfile(init_script)
        error('Initialization script not found: %s', init_script);
    end

    out_dir = char(opts.out_dir);
    if ~isfolder(out_dir)
        mkdir(out_dir);
    end

    cleanup_obj = onCleanup(@() cleanup_model_from_path(mdl_path)); %#ok<NASGU>

    if ~isempty(init_script)
        run_init_script(init_script, mdl_path);
        params = snapshot_parameters_from_base();
    else
        params = struct();
    end

    load_system(mdl_path);
    [~, mdl_name, ~] = fileparts(mdl_path);
    load_system('simulink');

    control_block = [mdl_name '/Constant2'];
    manual_switch_block = [mdl_name '/Manual Switch1'];
    sine_block = [mdl_name '/Sine Wave'];
    plant_block = [mdl_name '/Plant'];

    original_control_value = get_param(control_block, 'Value');
    if opts.force_control_zero
        set_param(control_block, 'Value', num2str(opts.u_voltage, '%.16g'));
    end

    tap_specs = {
        'top_F_h_ws', 'top_F_h_actual', [430 210 580 240], get_outport_handle(manual_switch_block, 1);
        'top_u_ws', 'top_u_actual', [430 330 580 360], get_outport_handle(control_block, 1);
        'top_x_s_ws', 'top_x_s_actual', [430 50 580 80], get_outport_handle(plant_block, 1);
        'top_x_m_ws', 'top_x_m_actual', [430 90 580 120], get_outport_handle(plant_block, 2);
        'top_Fe_ws', 'top_Fe_actual', [430 130 580 160], get_outport_handle(plant_block, 3);
        'top_x_mdot_ws', 'top_x_mdot_actual', [430 170 580 200], get_outport_handle(plant_block, 4);
        'top_x_sdot_ws', 'top_x_sdot_actual', [430 250 580 280], get_outport_handle(plant_block, 5)
    };

    for idx = 1:size(tap_specs, 1)
        add_top_level_tap(mdl_name, tap_specs{idx, 1}, tap_specs{idx, 2}, tap_specs{idx, 3}, tap_specs{idx, 4});
    end
    add_internal_to_workspace_tap(plant_block, 'top_K_e_ws', 'top_K_e_actual', [760 200 900 230], 'Sum4');
    add_internal_to_workspace_tap(plant_block, 'top_B_e_ws', 'top_B_e_actual', [760 240 900 270], 'Sum5');

    solver_settings = struct( ...
        'SolverType', get_param(mdl_name, 'SolverType'), ...
        'Solver', get_param(mdl_name, 'Solver'), ...
        'FixedStep', get_param(mdl_name, 'FixedStep'), ...
        'StartTime', get_param(mdl_name, 'StartTime'), ...
        'StopTime', get_param(mdl_name, 'StopTime'), ...
        'RelTol', get_param(mdl_name, 'RelTol'), ...
        'AbsTol', get_param(mdl_name, 'AbsTol'));

    source_settings = struct( ...
        'manual_switch_setting', get_param(manual_switch_block, 'sw'), ...
        'sine_amplitude', get_param(sine_block, 'Amplitude'), ...
        'sine_bias', get_param(sine_block, 'Bias'), ...
        'sine_frequency', get_param(sine_block, 'Frequency'), ...
        'sine_phase', get_param(sine_block, 'Phase'), ...
        'sine_sample_time', get_param(sine_block, 'SampleTime'));

    sim_out = sim(mdl_name, 'ReturnWorkspaceOutputs', 'on');

    raw = struct( ...
        'F_h', sim_out.get('top_F_h_actual'), ...
        'u', sim_out.get('top_u_actual'), ...
        'x_s', sim_out.get('top_x_s_actual'), ...
        'x_m', sim_out.get('top_x_m_actual'), ...
        'F_e', sim_out.get('top_Fe_actual'), ...
        'x_mdot', sim_out.get('top_x_mdot_actual'), ...
        'x_sdot', sim_out.get('top_x_sdot_actual'), ...
        'K_e', sim_out.get('top_K_e_actual'), ...
        'B_e', sim_out.get('top_B_e_actual'));

    data = collect_output_data(raw, opts.switch_time);
    results_full = build_results_table(data);
    results_20ms = results_full(1:choose_downsample_step(data.t, 0.02):height(results_full), :);

    metadata = struct( ...
        'source_model', mdl_path, ...
        'init_script', init_script, ...
        'parameter_source', ternary_string(isempty(init_script), 'saved_model_only', 'matlab_init_script'), ...
        'solver_type', solver_settings.SolverType, ...
        'solver', solver_settings.Solver, ...
        'fixed_step', solver_settings.FixedStep, ...
        'start_time', solver_settings.StartTime, ...
        'stop_time', solver_settings.StopTime, ...
        'manual_switch_setting', source_settings.manual_switch_setting, ...
        'sine_amplitude', source_settings.sine_amplitude, ...
        'sine_bias', source_settings.sine_bias, ...
        'sine_frequency', source_settings.sine_frequency, ...
        'sine_phase', source_settings.sine_phase, ...
        'sine_sample_time', source_settings.sine_sample_time, ...
        'control_value_original', original_control_value, ...
        'control_value_applied', get_param(control_block, 'Value'), ...
        'force_control_zero', logical(opts.force_control_zero), ...
        'switch_time_s', opts.switch_time, ...
        'samples_full', height(results_full), ...
        'samples_20ms', height(results_20ms));

    csv_path = fullfile(out_dir, 'matlab_top_level_io.csv');
    mat_path = fullfile(out_dir, 'matlab_top_level_debug.mat');
    scopes_path = fullfile(out_dir, 'matlab_top_level_scopes.png');
    error_scopes_path = fullfile(out_dir, 'matlab_top_level_error_scopes.png');
    summary_path = fullfile(out_dir, 'matlab_top_level_summary.txt');

    writetable(results_full, csv_path);
    save(mat_path, 'results_full', 'results_20ms', 'metadata', 'params', 'solver_settings', 'source_settings', 'data');
    plot_main_scopes(scopes_path, data, opts.switch_time);
    plot_error_scopes(error_scopes_path, data, opts.switch_time);
    write_summary(summary_path, data, metadata, solver_settings, source_settings);

    fprintf('MATLAB top-level as-run I/O export complete.\n');
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


function run_init_script(init_script, mdl_path)
    [script_folder, ~, script_ext] = fileparts(init_script);
    script_path = init_script;
    if isempty(script_ext)
        script_path = [init_script '.m'];
    end
    if ~isfile(script_path)
        mdl_folder = fileparts(mdl_path);
        alt_path = fullfile(mdl_folder, script_path);
        if isfile(alt_path)
            script_path = alt_path;
            script_folder = fileparts(script_path);
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
        'omega_v', 'zeta_v', 'K_v', 'K_e', 'B_e', 'P_md', 'k_h', 'B_h', ...
        'x_mE', 'x_sE'};
    params = struct();
    for idx = 1:numel(names)
        name = names{idx};
        if evalin('base', sprintf('exist(''%s'', ''var'')', name))
            params.(name) = evalin('base', name);
        end
    end
end


function add_top_level_tap(model, block_name, variable_name, position, source_outport)
    add_block('simulink/Sinks/To Workspace', [model '/' block_name], ...
        'VariableName', variable_name, ...
        'SaveFormat', 'Timeseries', ...
        'MaxDataPoints', 'inf', ...
        'Position', position);
    tap_port = get_param([model '/' block_name], 'PortHandles');
    add_line(model, source_outport, tap_port.Inport(1), 'autorouting', 'on');
end


function add_internal_to_workspace_tap(plant_block, block_name, variable_name, position, source_block_name)
    parent_model = get_param(plant_block, 'Parent');
    block_path = [plant_block '/' block_name];
    add_block('simulink/Sinks/To Workspace', block_path, ...
        'VariableName', variable_name, ...
        'SaveFormat', 'Timeseries', ...
        'MaxDataPoints', 'inf', ...
        'Position', position);
    source_handles = get_param([plant_block '/' source_block_name], 'PortHandles');
    tap_port = get_param(block_path, 'PortHandles');
    add_line(plant_block, source_handles.Outport(1), tap_port.Inport(1), 'autorouting', 'on');
    set_param(parent_model, 'Dirty', 'on');
end


function handle = get_outport_handle(block_path, outport_index)
    ports = get_param(block_path, 'PortHandles');
    handle = ports.Outport(outport_index);
end


function data = collect_output_data(raw, switch_time)
    data = struct();
    data.t = to_column(raw.x_m.Time);
    data.F_h = align_timeseries(raw.F_h, data.t);
    data.u = align_timeseries(raw.u, data.t);
    data.x_s = align_timeseries(raw.x_s, data.t);
    data.x_m = align_timeseries(raw.x_m, data.t);
    data.F_e = align_timeseries(raw.F_e, data.t);
    data.x_mdot = align_timeseries(raw.x_mdot, data.t);
    data.x_sdot = align_timeseries(raw.x_sdot, data.t);
    data.K_e = align_timeseries(raw.K_e, data.t);
    data.B_e = align_timeseries(raw.B_e, data.t);
    data.e = data.x_m - data.x_s;
    data.transparency_error = (data.F_e .* data.x_mdot) - (data.F_h .* data.x_sdot);
    data.env_label = strings(size(data.t));
    data.env_label(:) = "skin";
    data.env_label(data.t >= switch_time) = "fat";
end


function y = align_timeseries(ts_obj, ref_t)
    src_t = to_column(ts_obj.Time);
    src_y = to_column(ts_obj.Data);
    if numel(src_t) == 1
        y = repmat(src_y, size(ref_t));
        return;
    end
    if numel(src_t) == numel(ref_t) && all(abs(src_t - ref_t) < 1e-12)
        y = src_y;
        return;
    end
    y = interp1(src_t, src_y, ref_t, 'linear', 'extrap');
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
        data.K_e, ...
        data.B_e, ...
        data.env_label, ...
        'VariableNames', { ...
            't', 'F_h', 'u', 'x_m', 'x_s', 'x_mdot', 'x_sdot', 'F_e', 'e', 'transparency_error', 'K_e', 'B_e', 'env_label'});
end


function plot_main_scopes(out_path, data, switch_time)
    fig = figure('Visible', 'off', 'Position', [100 100 1400 1200]);
    tl = tiledlayout(7, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    plot_signal(nexttile(tl), data.t, data.F_h, switch_time, 'Actual F_h [model units]');
    plot_signal(nexttile(tl), data.t, data.u, switch_time, 'Actual u [V]');
    plot_signal(nexttile(tl), data.t, data.x_m, switch_time, 'Master position');
    plot_signal(nexttile(tl), data.t, data.x_s, switch_time, 'Slave position');
    plot_signal(nexttile(tl), data.t, data.x_mdot, switch_time, 'Master velocity');
    plot_signal(nexttile(tl), data.t, data.x_sdot, switch_time, 'Slave velocity');
    plot_signal(nexttile(tl), data.t, data.F_e, switch_time, 'Environment force');

    title(tl, 'MATLAB Top-Level As-Run I/O Scopes');
    xlabel(tl, 'Time [s]');
    exportgraphics(fig, out_path, 'Resolution', 160);
    close(fig);
end


function plot_error_scopes(out_path, data, switch_time)
    fig = figure('Visible', 'off', 'Position', [100 100 1400 750]);
    tl = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax1 = nexttile(tl);
    plot(ax1, data.t, data.e, 'LineWidth', 1.3, 'Color', [0.85 0.2 0.2]);
    yline(ax1, 0.0, 'Color', [0.1 0.1 0.1], 'LineWidth', 0.8);
    xline(ax1, switch_time, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.0);
    ylabel(ax1, 'Tracking error');
    grid(ax1, 'on');

    ax2 = nexttile(tl);
    plot(ax2, data.t, data.transparency_error, 'LineWidth', 1.3, 'Color', [0.2 0.35 0.85]);
    yline(ax2, 0.0, 'Color', [0.1 0.1 0.1], 'LineWidth', 0.8);
    xline(ax2, switch_time, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.0);
    ylabel(ax2, 'Transparency error');
    grid(ax2, 'on');

    ax3 = nexttile(tl);
    yyaxis(ax3, 'left');
    plot(ax3, data.t, data.K_e, 'LineWidth', 1.2, 'Color', [0.1 0.55 0.25]);
    ylabel(ax3, 'K_e');
    yyaxis(ax3, 'right');
    plot(ax3, data.t, data.B_e, 'LineWidth', 1.2, 'Color', [0.65 0.35 0.1]);
    ylabel(ax3, 'B_e');
    xline(ax3, switch_time, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.0);
    grid(ax3, 'on');

    title(tl, 'MATLAB Top-Level Tracking, Transparency, And Environment Switch');
    xlabel(tl, 'Time [s]');
    exportgraphics(fig, out_path, 'Resolution', 160);
    close(fig);
end


function write_summary(out_path, data, metadata, solver_settings, source_settings)
    fid = fopen(out_path, 'w');
    if fid == -1
        error('Failed to open summary file: %s', out_path);
    end
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid, 'MATLAB top-level as-run analysis\n');
    fprintf(fid, '===============================\n');
    fprintf(fid, 'source_model=%s\n', metadata.source_model);
    fprintf(fid, 'init_script=%s\n', metadata.init_script);
    fprintf(fid, 'parameter_source=%s\n', metadata.parameter_source);
    fprintf(fid, 'force_control_zero=%d\n', metadata.force_control_zero);
    fprintf(fid, 'control_value_original=%s\n', metadata.control_value_original);
    fprintf(fid, 'control_value_applied=%s\n', metadata.control_value_applied);
    fprintf(fid, 'switch_time_s=%.6f\n', metadata.switch_time_s);
    fprintf(fid, 'samples_full=%d\n', metadata.samples_full);
    fprintf(fid, 'samples_20ms=%d\n\n', metadata.samples_20ms);

    fprintf(fid, '[solver]\n');
    fprintf(fid, 'SolverType=%s\n', solver_settings.SolverType);
    fprintf(fid, 'Solver=%s\n', solver_settings.Solver);
    fprintf(fid, 'FixedStep=%s\n', solver_settings.FixedStep);
    fprintf(fid, 'StartTime=%s\n', solver_settings.StartTime);
    fprintf(fid, 'StopTime=%s\n', solver_settings.StopTime);
    fprintf(fid, 'RelTol=%s\n', solver_settings.RelTol);
    fprintf(fid, 'AbsTol=%s\n\n', solver_settings.AbsTol);

    fprintf(fid, '[top_level_source]\n');
    fprintf(fid, 'manual_switch_setting=%s\n', source_settings.manual_switch_setting);
    fprintf(fid, 'sine_amplitude=%s\n', source_settings.sine_amplitude);
    fprintf(fid, 'sine_bias=%s\n', source_settings.sine_bias);
    fprintf(fid, 'sine_frequency=%s\n', source_settings.sine_frequency);
    fprintf(fid, 'sine_phase=%s\n', source_settings.sine_phase);
    fprintf(fid, 'sine_sample_time=%s\n\n', source_settings.sine_sample_time);

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


function step = choose_downsample_step(t, target_dt)
    if numel(t) < 2
        step = 1;
        return;
    end
    native_dt = median(diff(t));
    step = max(1, round(target_dt / native_dt));
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


function cleanup_model_from_path(mdl_path)
    [~, mdl_name, ~] = fileparts(mdl_path);
    if bdIsLoaded(mdl_name)
        close_system(mdl_name, 0);
    end
end
