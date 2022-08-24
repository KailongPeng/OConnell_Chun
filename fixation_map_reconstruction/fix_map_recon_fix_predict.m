function fix_map_recon_fix_predict(roi)
% predict eye movements with direct fixation map reconstructions
% 50 GB memory load
%
% thomas oconnell

fprintf('Fixation Prediction from Fixation Map Reconstructions\n');

% params
subs = [2 3 5 6 9 11 13 14 15 16 18];
sub_vec = 1:numel(subs);
ROIs = {'V1','V2','V3','V4','LOC','PPA','FFA','OPA','RSC','IPS','FEF'};
im_size = [600 800];

% paths
cur_dir = pwd;
dir_ids = strfind(cur_dir,'/');
exp_path = cur_dir(1:dir_ids(end-1)-1);
recon_path = sprintf('%s/outputs/reconstructions',exp_path);
out_path = sprintf('%s/outputs/fixation_prediction',exp_path);
stim_file = sprintf('%s/data/salRecon_file_lists_MRI.mat',exp_path);
external_val_fix_file = sprintf('%s/data/scenes_exploration_fixation_coordinates.mat',exp_path);
addpath(genpath(sprintf('%s/scripts/utilities',exp_path)));

% load image lists
stim_list = load(stim_file);
fnames = unique(stim_list.salRecon_lists.files(1,:,:));
im_names = cellfun(@(x) x(1:end-4),fnames,'Un',0);

% generate permutation indices
rng('shuffle');
permutations = 1000;
for perm = 1:permutations
    perm_inds{perm} = randperm(numel(im_names));
end

% Load fixation map reconstructions
fprintf('Loading Fixation Map Reconstructions\n');
recons = load(sprintf('%s/fixation_map_reconstructions_%s.mat',recon_path,ROIs{roi}));

% Load Internal Validation Fixation Data
fprintf('Loading Fixation Data (Internal Validation)\n');
internal_fix_maps = zeros(numel(subs),numel(im_names),im_size(1),im_size(2));
params.exp_path = exp_path;
params.fixs_before = 2000; % use all fixations before (ms)
for s = 1:numel(subs)
    [internal_fix_inds(s,:),~,~,~] = load_fixation_data_salRecon(s,im_names,params);
    % make fixation maps
    for im = 1:size(internal_fix_inds,2)
        cur_map = zeros(im_size);
        if ~isempty(internal_fix_inds{s,im})
            cur_map(internal_fix_inds{s,im}) = 1;
        else
            cur_map = NaN(im_size);
        end
        internal_fix_maps(s,im,:,:) = cur_map;
    end
end
internal_fix_counts = cellfun(@numel,internal_fix_inds);

% Within-Subject Validation
fprintf('Within-Subject Validation\n');
nss_calc = recons.recons_fix_map_all_subs .* internal_fix_maps;
nss_calc = reshape(nss_calc,numel(subs),numel(im_names),prod(im_size));
fix_pred_out.wiS_nss = sum(nss_calc,3) ./ internal_fix_counts;
clear nss_calc;
% Permutation Tests
run_time = NaN;
for perm = 1:permutations
    msg = sprintf('Permutation = %d, Previous Run Time = %2.2f',perm,run_time);
    fprintf(msg);
    tic
    nss_calc = recons.recons_fix_map_all_subs(:,perm_inds{perm},:,:) .* internal_fix_maps;
    nss_calc = reshape(nss_calc,numel(subs),numel(im_names),prod(im_size));
    fix_pred_out.wiS_nss_perm(perm,:) = nanmean(sum(nss_calc,3) ./ internal_fix_counts,2);
    clear nss_calc;
    run_time = toc;
    if perm~=permutations
        fprintf(repmat('\b',1,numel(msg)));
    end
end
fprintf('\n');

% Internal Validation
fprintf('Internal Validation\n');
for s = 1:numel(subs)
    fprintf('Subject %d\n',subs(s));
    % extract fixation for current subject
    fix_maps_cur = squeeze(internal_fix_maps(s,:,:,:));
    % extract group-average reconstructions
    gr_av_recons_minus_sub = squeeze(nanmean(recons.recons_fix_map_all_subs(sub_vec~=s,:,:,:)));
    gr_av_recons_minus_sub = reshape(zscore(reshape(gr_av_recons_minus_sub,numel(im_names),prod(im_size)),[],2),[numel(im_names) im_size]);
    % calculate nss
    nss_calc = gr_av_recons_minus_sub .* fix_maps_cur;
    nss_calc = reshape(nss_calc,numel(im_names),prod(im_size));
    fix_pred_out.internal_nss(s,:) = sum(nss_calc,2)./internal_fix_counts(s,:)';
    clear nss_calc;
    % Permutation Tests
    run_time = NaN;
    for perm = 1:permutations
        msg = sprintf('Permutation = %d, Previous Run Time = %2.2f',perm,run_time);
        fprintf(msg);
        tic
        nss_calc = gr_av_recons_minus_sub(perm_inds{perm},:,:) .* fix_maps_cur;
        nss_calc = reshape(nss_calc,numel(im_names),prod(im_size));
        fix_pred_out.internal_nss_perm(perm,s) = nanmean(sum(nss_calc,2) ./ internal_fix_counts(s,:)');
        run_time = toc;
        if perm~=permutations
            fprintf(repmat('\b',1,numel(msg)));
        end
    end
    fprintf('\n'); clear gr_av_recons_minus_sub nss_calc;
end
clear internal_fix_inds internal_fix_maps internal_fix_counts;

% Load External Validation Fixation Data
% O'Connell & Walther 2015
fprintf('Loading Fixation Data (External Validation)\n');
external_fix_maps = zeros(numel(subs),numel(im_names),im_size(1),im_size(2));
external_fix_dat = load(external_val_fix_file);
external_fix_inds = external_fix_dat.fix_inds;
% make fixation maps
for s = 1:size(external_fix_inds,1)
    for im = 1:size(external_fix_inds,2)
        cur_map = zeros(im_size);
        if ~isempty(external_fix_inds{s,im})
            cur_map(external_fix_inds{s,im}) = 1;
        else
            cur_map = NaN(im_size);
        end
        external_fix_maps(s,im,:,:) = cur_map;
    end
end
external_fix_counts = cellfun(@numel,external_fix_inds);
clear external_fix_dat external_fix_inds;

% External Validation
fprintf('External Validation\n');
% average reconstructions across all subjects
gr_av_recons = squeeze(nanmean(recons.recons_fix_map_all_subs));    
gr_av_recons = reshape(zscore(reshape(gr_av_recons,numel(im_names),prod(im_size)),[],2),[numel(im_names) im_size]);
% calculate nss
nss_calc = bsxfun(@times,shiftdim(gr_av_recons,-1),external_fix_maps);
nss_calc = reshape(nss_calc,size(external_fix_maps,1),numel(im_names),prod(im_size));
fix_pred_out.external_nss = sum(nss_calc,3) ./ external_fix_counts;
clear nss_calc;
% Permutation Tests
run_time = NaN;
for perm = 1:permutations
    msg = sprintf('Permutation = %d, Previous Run Time = %2.2f',perm,run_time);
    fprintf(msg);
    tic
    nss_calc = bsxfun(@times,shiftdim(gr_av_recons(perm_inds{perm},:,:),-1),external_fix_maps);
    nss_calc = reshape(nss_calc,size(external_fix_maps,1),numel(im_names),prod(im_size));
    fix_pred_out.external_nss_perm(perm,:) = nanmean(sum(nss_calc,3) ./ external_fix_counts,2);
    clear nss_calc;
    run_time = toc;
    if perm~=permutations
        fprintf(repmat('\b',1,numel(msg)));
    end
end
fprintf('\n'); clear gr_av_recons;

% save output
save(sprintf('%s/fixation_prediction_results_fixMapRecon_%s.mat',...
     out_path,ROIs{roi}),'fix_pred_out','-v7.3');

