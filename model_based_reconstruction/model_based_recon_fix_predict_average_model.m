function model_based_recon_fix_predict_average_model(index)
% predit eye movements with model-based reconstructions
% 50 GB memory load
%
% thomas oconnell

fprintf('Fixation Prediction from Model-Based Reconstructions\n');

% params
subs = [2 3 5 6 9 11 13 14 15 16 18];
sub_vec = 1:numel(subs);
ROIs = {'V1','V2','V3','V4','LOC','PPA','FFA','OPA','RSC','IPS','FEF'};
%cost_functions = {'places365','ILSVRC','face'};
cost_functions = {'places365'};
im_size = [600 800];

[A,B] = meshgrid([1:numel(cost_functions)],[1:numel(ROIs)]);
c=cat(2,A',B');
d=reshape(c,[],2);
cost = d(index,1)
roi = d(index,2)

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

fprintf('Training Set: %s\n',cost_functions{cost});

% load model-based reconstructions
fprintf('Loading Model-Based Reconstructions\n');
recons = load(sprintf('%s/model_based_reconstructions_%s_%s.mat',...
              recon_path,cost_functions{cost},ROIs{roi}),'recons_average_base','recons_average_total');
    
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
% Base
nss_calc = recons.recons_average_base .* internal_fix_maps;
nss_calc = reshape(nss_calc,numel(subs),numel(im_names),prod(im_size));
fix_pred_out.base.wiS_nss = sum(nss_calc,3) ./ internal_fix_counts;
clear nss_calc;
% Total (Smoothing + Center-Bias Correction)
nss_calc = recons.recons_average_total .* internal_fix_maps;
nss_calc = reshape(nss_calc,numel(subs),numel(im_names),prod(im_size));
fix_pred_out.total.wiS_nss = sum(nss_calc,3) ./ internal_fix_counts;
clear nss_calc;
% Permutation Tests
run_time = NaN;
for perm = 1:permutations
   msg = sprintf('Permutation = %d, Previous Run Time = %2.2f',perm,run_time);
   fprintf(msg);
   tic
   % Base
   nss_calc = recons.recons_average_base(:,perm_inds{perm},:,:) .* internal_fix_maps;
   nss_calc = reshape(nss_calc,numel(subs),numel(im_names),prod(im_size));
   fix_pred_out.base.wiS_nss_perm(perm,:) = nanmean(sum(nss_calc,3) ./ internal_fix_counts,2);
   clear nss_calc;
   % Total (Smoothing + Center-Bias Correction)
   nss_calc = recons.recons_average_total(:,perm_inds{perm},:,:) .* internal_fix_maps;
   nss_calc = reshape(nss_calc,numel(subs),numel(im_names),prod(im_size));
   fix_pred_out.total.wiS_nss_perm(perm,:) = nanmean(sum(nss_calc,3) ./ internal_fix_counts,2);
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
    % Base
    fprintf('Base Model\n');
    gr_av_recons_minus_sub.base = squeeze(nanmean(recons.recons_average_base(sub_vec~=s,:,:,:)));
    gr_av_recons_minus_sub.base = reshape(zscore(reshape(gr_av_recons_minus_sub.base,numel(im_names),prod(im_size)),[],2),[numel(im_names) im_size]);
    nss_calc = gr_av_recons_minus_sub.base .* fix_maps_cur;
    nss_calc = reshape(nss_calc,numel(im_names),prod(im_size));
    fix_pred_out.base.internal_nss(s,:) = sum(nss_calc,2) ./ internal_fix_counts(s,:)';
   % Permutation Tests
   run_time = NaN;
   for perm = 1:permutations
       msg = sprintf('Permutation = %d, Previous Run Time = %2.2f',perm,run_time);
       fprintf(msg);
       tic
       nss_calc = gr_av_recons_minus_sub.base(perm_inds{perm},:,:) .* fix_maps_cur;
       nss_calc = reshape(nss_calc,numel(im_names),prod(im_size));
       fix_pred_out.base.internal_nss_perm(perm,s) = nanmean(sum(nss_calc,2) ./ internal_fix_counts(s,:)');
       run_time = toc;
       if perm~=permutations
           fprintf(repmat('\b',1,numel(msg)));
       end
   end
    fprintf('\n'); clear gr_av_recons_minus_sub nss_calc;
    % Total (Smoothing + Center-Bias Correction)
    fprintf('Smoothed + Center-Bias Corrected Model\n');
    gr_av_recons_minus_sub.total = squeeze(nanmean(recons.recons_average_total(sub_vec~=s,:,:,:)));
    gr_av_recons_minus_sub.total = reshape(zscore(reshape(gr_av_recons_minus_sub.total,numel(im_names),prod(im_size)),[],2),[numel(im_names) im_size]);        
    nss_calc = gr_av_recons_minus_sub.total .* fix_maps_cur;
    nss_calc = reshape(nss_calc,numel(im_names),prod(im_size));
    fix_pred_out.total.internal_nss(s,:) = sum(nss_calc,2) ./ internal_fix_counts(s,:)';
   % Permutation Tests
   run_time = NaN;
   for perm = 1:permutations
       msg = sprintf('Permutation = %d, Previous Run Time = %2.2f',perm,run_time);
       fprintf(msg);
       tic
       nss_calc = gr_av_recons_minus_sub.total(perm_inds{perm},:,:) .* fix_maps_cur;
       nss_calc = reshape(nss_calc,numel(im_names),prod(im_size));
       fix_pred_out.total.internal_nss_perm(perm,s) = nanmean(sum(nss_calc,2) ./ internal_fix_counts(s,:)');
       run_time = toc;
       if perm~=permutations
           fprintf(repmat('\b',1,numel(msg)));
       end
   end
    fprintf('\n'); clear gr_av_recons_minus_sub fix_maps_cur nss_calc;
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

% External Validation
fprintf('External Validation\n');
% Base
fprintf('Base Model\n');
gr_av_recons.base = squeeze(nanmean(recons.recons_average_base));
gr_av_recons.base = reshape(zscore(reshape(gr_av_recons.base,numel(im_names),prod(im_size)),[],2),[numel(im_names) im_size]);
nss_calc = bsxfun(@times,shiftdim(gr_av_recons.base,-1),external_fix_maps);
nss_calc = reshape(nss_calc,size(external_fix_maps,1),numel(im_names),prod(im_size));
fix_pred_out.base.external_nss = sum(nss_calc,3) ./ external_fix_counts;
clear nss_calc;
% Permutation Tests
run_time = NaN;
for perm = 1:permutations
   msg = sprintf('Permutation = %d, Previous Run Time = %2.2f',perm,run_time);
   fprintf(msg);
   tic
   nss_calc = bsxfun(@times,shiftdim(gr_av_recons.base(perm_inds{perm},:,:),-1),external_fix_maps);
   nss_calc = reshape(nss_calc,size(external_fix_maps,1),numel(im_names),prod(im_size));
   fix_pred_out.base.external_nss_perm(perm,:) = nanmean(sum(nss_calc,3) ./ external_fix_counts,2);
   clear nss_calc;
   run_time = toc;
   if perm~=permutations
       fprintf(repmat('\b',1,numel(msg)));
   end
end
fprintf('\n'); clear gr_av_recons;
% Total
fprintf('Total Model\n');
gr_av_recons.total = squeeze(nanmean(recons.recons_average_total));
gr_av_recons.total = reshape(zscore(reshape(gr_av_recons.total,numel(im_names),prod(im_size)),[],2),[numel(im_names) im_size]);        
nss_calc = bsxfun(@times,shiftdim(gr_av_recons.total,-1),external_fix_maps);
nss_calc = reshape(nss_calc,size(external_fix_maps,1),numel(im_names),prod(im_size));
fix_pred_out.total.external_nss = sum(nss_calc,3) ./ external_fix_counts;
clear nss_calc;
% Permutation Tests
run_time = NaN;
for perm = 1:permutations
   msg = sprintf('Permutation = %d, Previous Run Time = %2.2f',perm,run_time);
   fprintf(msg);
   tic
   nss_calc = bsxfun(@times,shiftdim(gr_av_recons.total(perm_inds{perm},:,:),-1),external_fix_maps);
   nss_calc = reshape(nss_calc,size(external_fix_maps,1),numel(im_names),prod(im_size));
   fix_pred_out.total.external_nss_perm(perm,:) = nanmean(sum(nss_calc,3) ./ external_fix_counts,2);
   clear nss_calc;
   run_time = toc;
   if perm~=permutations
       fprintf(repmat('\b',1,numel(msg)));
   end
end
fprintf('\n'); clear gr_av_recons;

% save output
save(sprintf('%s/fixation_prediction_results_%s_%s.mat',...
     out_path,cost_functions{cost},ROIs{roi}),'fix_pred_out','-v7.3');
clear recons fix_pred_out;
