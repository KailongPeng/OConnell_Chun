function fixation_map_direct_decoding(index)
% 直接解码的固定模式 - salRecon_revision.   direct decoding of fixation patterns - salRecon_revision
% 25 GB memory load
%
% thomas oconnell

% params
subs = [2 3 5 6 9 11 13 14 15 16 18];
ROIs = {'V1','V2','V3','V4','LOC','PPA','FFA','OPA','RSC','IPS','FEF'};
trials=24;
im_size = [600 800];

[A,B] = meshgrid([1:11],[1:11]);
c=cat(2,A',B');
d=reshape(c,[],2);
s = d(index,1)
roi = d(index,2)

% paths
cur_dir = pwd;
dir_ids = strfind(cur_dir,'/');
exp_path = cur_dir(1:dir_ids(end-1)-1);
stim_file = sprintf('%s/data/salRecon_file_lists_MRI.mat',exp_path);
fix_file = sprintf('%s/data/all_fixation_maps_171017.mat',exp_path);
out_path = sprintf('%s/outputs/model_aligned_bold_activity',exp_path);
addpath(genpath(sprintf('%s/scripts/utilities',exp_path)));

% 被试run数据.  Subject run data
runs = {[1 2 3 4 5 6 7 8 9 10 11 12],...
        [1 2 3 4 5 6 7 8 9 10 11 12],...
        [1 2 3 4 5 6 7 8 9 10 11 12],...
        [1 2 3 4 5 6 7 8 9 10 11 12],...
        [1 2 3 4 5 6 7 8 9 10 11 12],...
        [1 2 3 4 5 6 7 8 9 10],...
        [1 2 3 4 5 6 7 8 9 10 11 12],...
        [1 2 3 4 5 6 7 8 9 10 11 12],...
        [1 2 3 4 5 6 7 8 9 10 11 12],...
        [1 2 3 4 5 6 7 8 9 10 11 12],...
        [1 2 3 4 5 6 7 9 10 11 12]};
num_runs = cellfun(@numel,runs);

% 加载图像列表.  load image lists
fprintf('Load Image List\n');
stim_list = load(stim_file);
fnames = unique(stim_list.salRecon_lists.files(1,:,:));
im_names = cellfun(@(x) x(1:end-4),fnames,'Un',0);

% 从当前主题中提取图像列表.  extract image list from current subject
fold_inds = repmat(1:num_runs(s),trials,1);
fold_inds = fold_inds(:);
run_inds = repmat(runs{s},trials,1);
run_inds = run_inds(:);
sub_file_list = squeeze(stim_list.salRecon_lists.files(subs(s),runs{s},:))';
sub_file_list = sub_file_list(:);
sub_im_list = cellfun(@(x) x(1:end-4),sub_file_list,'Un',0);

% 制作重复的标签.  make repetition labels
rep_labs = NaN(size(sub_im_list));
for im = 1:numel(im_names)
    cur_inds = find(strcmp(sub_im_list,im_names{im}));
    if isempty(cur_inds)
        continue
    end
    rep_labs(cur_inds(1)) = 1;
    rep_labs(cur_inds(2)) = 2;
end

% 负载固定图.  load fixation maps
fprintf('Load Fixation Maps\n');
all_fix_maps = load(fix_file);
sub_fix_maps = all_fix_maps.FDMs{s,3}; % index 3, sigma SD = 20pi, determined via cross-validation
clear all_fix_maps;

% load BOLD activity
fprintf('Load BOLD Activity\n');
bold_activity = load_masked_surface_data_final(sprintf('sub%d',subs(s)),roi,exp_path,num_runs(s),1);

% create design matrix
for im = 1:numel(sub_im_list)
    sub_inds(im) = find(cellfun(@(x) strcmp(x,sub_im_list{im}),im_names));
end
fix_map_feats = reshape(sub_fix_maps(sub_inds,:,:),numel(sub_im_list),im_size(1)*im_size(2));

% 固定模式的直接解码.  Direct decoding of fixation patterns
fprintf('Decoding Fixation Maps...');
fix_map_recons = [];
for fold = 1:numel(unique(fold_inds))
    fprintf('%d..',fold);
    % 定义训练/测试指数.  define train/test indices
    tr_inds = fold_inds~=fold;
    te_inds = fold_inds==fold;
    % 降维--固定图.  dimensionality reduction - fixation maps
    unique_fix_map_feats_tr_set = fix_map_feats(intersect(find(rep_labs==1),find(tr_inds)),:);
    pca_transform_fix = pca(unique_fix_map_feats_tr_set);
    fix_map_feats_comp_train = fix_map_feats(tr_inds,:) * pca_transform_fix;
    train_inds_to_remove = any(isnan(fix_map_feats_comp_train),2);
    fix_map_feats_comp_train(train_inds_to_remove,:) = []; % remove fix maps without any fixations
    fix_map_feats_comp_test = fix_map_feats(te_inds,:) * pca_transform_fix;
    % 降维 - BOLD.  dimensionality reduction - BOLD
    [pca_transform_bold,bold_activity_comp_train,~] = pca(bold_activity(tr_inds,:));
    bold_activity_comp_test = bold_activity(te_inds,:) * pca_transform_bold;
    if ~isempty(train_inds_to_remove)
        bold_activity_comp_train(train_inds_to_remove,:) = [];
    end
    % 学习BOLD > 固定图转换 learn BOLD > fixation map transformation
    [~,~,~,~,weights] = plsregress(bold_activity_comp_train,fix_map_feats_comp_train,130);
    brain_to_fix_transformation = weights(2:end,:);
    % 转化测试BOLD活动在固定图成分空间中的位置 transform test BOLD activity in fixation map component space
    bold_decoded_fix_components = bold_activity_comp_test*brain_to_fix_transformation;
    % 将解码后的固定图成分投射到完整的固定图中。  project decoded fixation map components into full fixation map
    fix_map_recons = [fix_map_recons; bold_decoded_fix_components * pca_transform_fix'];
end
fprintf('\n');
% 重塑成图像  reshape into images
fix_map_recons = reshape(fix_map_recons,[size(fix_map_recons,1) im_size]);

% 保存固定图的重构  save fixation map reconstructions
fprintf('Saving Reconstructed Fixation Maps\n');
save(sprintf('%s/sub%d_%s_fixation_map_aligned_bold.mat',out_path,subs(s),ROIs{roi}),...
     'fix_map_recons','run_inds','fold_inds','rep_labs','sub_file_list','-v7.3');



