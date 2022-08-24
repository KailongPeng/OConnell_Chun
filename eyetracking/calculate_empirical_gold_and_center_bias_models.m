function calculate_empirical_gold_and_center_bias_models
%
% thomas oconnell

% params
subs = [2 3 5 6 9 11 13 14 15 16 18];
im_size = [600 800];

% paths
cur_dir = pwd;
dir_ids = strfind(cur_dir,'/');
exp_path = cur_dir(1:dir_ids(end-1)-1);
stim_file = sprintf('%s/data/salRecon_file_lists_MRI.mat',exp_path);
external_val_fix_file = sprintf('%s/data/scenes_exploration_fixation_coordinates.mat',exp_path);
out_file = sprintf('%s/outputs/computational_model_files/gold_standard_and_baseline_maps.mat',exp_path);
addpath(genpath(sprintf('%s/scripts/utilities',exp_path)));

% load image lists
stim_list = load(stim_file);
fnames = unique(stim_list.salRecon_lists.files(1,:,:));
im_names = cellfun(@(x) x(1:end-4),fnames,'Un',0);
im_vec = 1:numel(im_names);

% Load Internal Validation Fixation Data
fprintf('Loading Fixation Data (Internal Validation)\n');
params.exp_path = exp_path;
params.fixs_before = 2000; % use all fixations before (ms)
for s = 1:numel(subs)
    [fix_inds{1}(s,:),~,~,~] = load_fixation_data_salRecon(s,im_names,params);
    % make fixation maps (no smoothing)
    for im = 1:size(fix_inds{1},2)
        cur_map = zeros(im_size);
        if ~isempty(fix_inds{1}{s,im})
            cur_map(fix_inds{1}{s,im}) = 1;
        else
            cur_map = NaN(im_size);
        end
        fix_maps{1}(s,im,:,:) = cur_map;
    end
end

% Load External Validation Fixation Data
% O'Connell & Walther 2015
fprintf('Loading Fixation Data (External Validation)\n');
external_fix_dat = load(external_val_fix_file);
fix_inds{2} = external_fix_dat.fix_inds;
% make fixation maps (no smoothing)
for s = 1:size(fix_inds{2},1)
    for im = 1:size(fix_inds{2},2)
        cur_map = zeros(im_size);
        if ~isempty(fix_inds{2}{s,im})
            cur_map(fix_inds{2}{s,im}) = 1;
        else
            cur_map = NaN(im_size);
        end
        fix_maps{2}(s,im,:,:) = cur_map;
    end
end

% define gaussian kernel
sigma = 20; % determined on gold standard model using gridsearch
gauss_kernel = fspecial('gaussian',6*sigma,sigma);

% loop over internal/external validation
fprintf('Compute Gold Standard and Baseline Maps\n');
set_sw = [2 1];
gold_standard_maps = cell(2,1);
baseline_maps = cell(2,1);
for set = 1:2
    % define gold-standard maps
    gold_standard_fix_points = squeeze(nansum(fix_maps{set_sw(set)}));
    % define baseline maps
    baseline_fix_points = NaN(size(gold_standard_fix_points));
    for im = 1:size(gold_standard_fix_points,1)
        baseline_fix_points(im,:,:) = squeeze(sum(gold_standard_fix_points(im_vec~=im,:,:)));
    end
    % smoothing + normalization
    gold_standard_maps{set} = NaN(size(gold_standard_fix_points));
    baseline_maps{set} = NaN(size(baseline_fix_points));
    for im = 1:size(gold_standard_fix_points,1)
        % gold standard
        cur_map = imfilter(squeeze(gold_standard_fix_points(im,:,:)),gauss_kernel,'conv');
        gold_standard_maps{set}(im,:,:) = reshape(zscore(cur_map(:)),im_size);
        % baseline
        cur_map = imfilter(squeeze(baseline_fix_points(im,:,:)),gauss_kernel,'conv');
        baseline_maps{set}(im,:,:) = reshape(zscore(cur_map(:)),im_size);
    end
end
    
% save maps
data_legend = {'Maps for Internal Validation prediction defined on External Validation data',...
       'Maps for External Validation prediction defined on Internal Validation data'};
save(out_file,'data_legend','gold_standard_maps','baseline_maps','-v7.3');
    
    
