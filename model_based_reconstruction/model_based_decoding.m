function model_based_decoding(index)
% 主解码脚本 - salRecon_revision.  master decoding script - salRecon_revision
% 25 GB memory load
%
% thomas oconnell

% params
subs = [2 3 5 6 9 11 13 14 15 16 18];
layers = {'pool1','pool2','pool3','pool4','pool5'};
layer_dims = [64 112 112;
              128 56 56;
              256 28 28;
              512 14 14;
              512 7 7];
ROIs = {'V1','V2','V3','V4','LOC','PPA','FFA','OPA','RSC','IPS','FEF'};
cost_functions = {'places365'};
% cost_functions = {'places365','ILSVRC','face'};
trials=24;

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
feat_path = sprintf('%s/vgg16_activity_all_models',exp_path);
out_path = sprintf('%s/outputs/model_aligned_bold_activity',exp_path);
addpath(genpath(sprintf('%s/scripts/utilities',exp_path)));

% 被试run数据  Subject run data
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

% load image lists
fprintf('Load Image List\n');
stim_list = load(stim_file);
fnames = unique(stim_list.salRecon_lists.files(1,:,:));
im_names = cellfun(@(x) x(1:end-4),fnames,'Un',0);

% 从当前主题中提取图像列表  extract image list from current subject
fold_inds = repmat(1:num_runs(s),trials,1);
fold_inds = fold_inds(:);
run_inds = repmat(runs{s},trials,1);
run_inds = run_inds(:);
sub_file_list = squeeze(stim_list.salRecon_lists.files(subs(s),runs{s},:))';
sub_file_list = sub_file_list(:);
sub_im_list = cellfun(@(x) x(1:end-4),sub_file_list,'Un',0);

% 制作重复的标签  make repetition labels
rep_labs = NaN(size(sub_im_list));
for im = 1:numel(im_names)
    cur_inds = find(strcmp(sub_im_list,im_names{im}));
    if isempty(cur_inds)
        continue
    end
    rep_labs(cur_inds(1)) = 1;
    rep_labs(cur_inds(2)) = 2;
end

% 保存被试元数据  save subject metadata
if roi==1
    out_table = table(sub_file_list, run_inds, fold_inds, rep_labs, ...
                      'variablenames',{'filename','run','fold','rep'});
    writetable(out_table,sprintf('%s/sub%d_metada.csv',out_path,subs(s)));
end

% 负载BOLD活动  load BOLD activity
fprintf('Load BOLD Activity\n');
bold_activity = load_masked_surface_data_final(sprintf('sub%d',subs(s)),roi,exp_path,num_runs(s),1);

% 循环计算成本函数  loop over cost functions
for cost = 1:numel(cost_functions)
    fprintf('COST FUNCTION: %s\n',cost_functions{cost});
    
    % 层层循环  loop over layers
    for layer = 1:numel(layers)
        fprintf('LAYER: %s\n',layers{layer});
        
        % 加载vgg活动  load vgg activity
        fprintf('Load VGG16 Activity\n');
        vgg_in = load(sprintf('%s/%s_%s_activity.mat',feat_path,cost_functions{cost},layers{layer}));
        vgg_act_cur = vgg_in.all_layer_feats;
        vgg_act_cur = zscore(vgg_act_cur,0,2);
        % 创建设计矩阵  create design mat
        for im = 1:numel(sub_im_list)
            cur_ind = find(cellfun(@(x) strcmp(x,sub_im_list{im}),im_names));
            if isempty(cur_ind)
                continue
            end
            vgg_activity(im,:) = vgg_act_cur(cur_ind,:);
        end
        clear vgg_act_cur vgg_in;
        
        % 基于模型的解码  Model-based decoding
        fprintf('Model-Based Decoding...');
        model_aligned_bold = [];
        for fold = 1:numel(unique(fold_inds))
            fprintf('%d..',fold);
            % 定义训练/测试指数 define train/test indices
            tr_inds = fold_inds~=fold;
            te_inds = fold_inds==fold;
            % 降维 - BOLD  dimensionality reduction - BOLD
            [pca_transform_bold,bold_activity_comp_train,~] = pca(bold_activity(tr_inds,:));
            bold_activity_comp_test = bold_activity(te_inds,:) * pca_transform_bold;
            % 降维 - VGG16  dimensionality reduction - VGG16
            unique_vgg_activity_tr_set = vgg_activity(intersect(find(rep_labs==1),find(tr_inds)),:);
            pca_transform_vgg = pca(unique_vgg_activity_tr_set);
            vgg_activity_comp_train = vgg_activity(tr_inds,:) * pca_transform_vgg;
            vgg_activity_comp_test = vgg_activity(te_inds,:) * pca_transform_vgg;
            % 学习BOLD > VGG16改造  learn BOLD > VGG16 transformation
            [~,~,~,~,weights] = plsregress(bold_activity_comp_train,vgg_activity_comp_train,130);
            brain_to_model_transformation = weights(2:end,:);
            % 将测试的BOLD活动转化为VGG16成分空间 transform test BOLD activity into VGG16 component space
            bold_decoded_vgg_components=bold_activity_comp_test*brain_to_model_transformation;
            % 将解码的VGG16组件投射到全层活动空间中 project decoded VGG16 components into full layer activity space
            model_aligned_bold = [model_aligned_bold; bold_decoded_vgg_components * pca_transform_vgg'];
        end
        fprintf('\n');
        % 保存与模型相一致的大胆活动  save model-aligned bold activity
        fprintf('Saving VGG-Aligned BOLD Activity\n');
        save(sprintf('%s/sub%d_%s_vgg_%s_%s_aligned_bold.mat',out_path,subs(s),ROIs{roi},cost_functions{cost},layers{layer}),...
             'model_aligned_bold','run_inds','fold_inds','rep_labs','sub_file_list','-v7.3');
        clear vgg_activity;
    end
end

