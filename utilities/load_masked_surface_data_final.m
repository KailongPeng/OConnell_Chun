function activity = load_masked_surface_data_final(sub,roi,exp_path,num_runs,norm_flag)
% Load surface activity (masked) - salRecon
% Inputs
% -sub: string with subject ID
% -roi: roi index (1:11)
% -exp_path: path to experiment directory
% -num_runs: # of runs for current subject
% -norm_flag: 1 = zscore
%
% thomas oconnell

% params
hemis = {'lh','rh'};
run_labs = {'01','02','03','04','05','06','07','08','09','10','11','12'};
run_labs = run_labs(1:num_runs);
trials = 24;
tr_run_inds = zeros(1,279); tr_run_inds(6:11:259)=1:trials;  
tr_run_inds = find(tr_run_inds);

% initialize array for functional data
fprintf('Loading Functional Data...Surface...salRecon\n');

for hem = 1:numel(hemis)
    fprintf('%s...',hemis{hem});
    count=1;
    for run = 1:1:numel(run_labs)
        fprintf('Run %d..',run);
        % load matrix for current run
        fdat_cur = dlmread(sprintf('%s/data/%s/fMRI/%s.%s.r%s.patterns.masked',...
                    exp_path,sub,sub,hemis{hem},run_labs{run}));
        if run==1
            node_inds = fdat_cur(:,1);
        end
        fdat_cur = fdat_cur(:,2:end);
        % extract trial activity
        for tr = 1:trials
            % average data from TR 5 to TR 8 for each trial
            fdat(:,count) = mean(fdat_cur(:,tr_run_inds(tr)+4:tr_run_inds(tr)+7),2);
            count=count+1;
        end
    end
    % load ROI indices
    node_inds_all = NaN(size(node_inds));
    roi_inds_all = NaN(size(node_inds));
    fid = fopen(sprintf('%s/data/%s/fMRI/func_rois_final.%s.1D.roi',...
                exp_path,sub,hemis{hem}),'rt');
    count=1;
    tline = fgetl(fid);
    while ischar(tline)
        if ~isempty(tline) && ~strcmp(tline(1),'#')
            dline = strsplit(tline,' ');
            node_inds_all(count) = str2num(dline{2});
            roi_inds_all(count) = str2num(dline{3});
            count=count+1;
        end
        tline = fgetl(fid);
    end
    fclose(fid);
    % find node indices for current roi
    cur_node_inds = node_inds_all(roi_inds_all==roi);
    % find current node indices in fdat node inds
    cur_inds =arrayfun(@(x) find(node_inds==x),cur_node_inds);
    % extract activity for 
    activity{hem} = fdat(cur_inds,:);
    fprintf('\n'); clear fdat_cur fdat;
end

% concatenate across hemispheres
activity = [activity{1}' activity{2}'];

% zscore
if norm_flag==1
    activity = zscore(activity,0,2);
end
