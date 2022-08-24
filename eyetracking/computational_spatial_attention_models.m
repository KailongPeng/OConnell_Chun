function computational_spatial_attention_models
% 
% thomas oconnell

fprintf('Computational Spatial Attention Models\n');

% params
subs = [2 3 5 6 9 11 13 14 15 16 18];
layers = {'pool1','pool2','pool3','pool4','pool5'};
cost_functions = {'places365','ILSVRC','face'};
validation_types = {'Internal Validation','External Validation'};
comp_model_types = {'Base Model','Smoothed','C-B Corrected','Smoothed & C-B Corrected'};
bench_model_types = {'MIT Center','Our Center','Gold-Standard'};
im_size = [600 800];

% paths
cur_dir = pwd;
dir_ids = strfind(cur_dir,'/');
exp_path = cur_dir(1:dir_ids(end-1)-1);
out_path = sprintf('%s/outputs/computational_model_files',exp_path);
stim_file = sprintf('%s/data/salRecon_file_lists_MRI.mat',exp_path);
external_val_fix_file = sprintf('%s/data/scenes_exploration_fixation_coordinates.mat',exp_path);
addpath(genpath(sprintf('%s/scripts/utilities',exp_path)));

% load image lists
stim_list = load(stim_file);
fnames = unique(stim_list.salRecon_lists.files(1,:,:));
im_names = cellfun(@(x) x(1:end-4),fnames,'Un',0);

% generate permutation indicess
rng('shuffle');
permutations = 1000;
for perm = 1:permutations
    perm_inds{perm} = randperm(numel(im_names));
end

% MIT saliency benchmark center model
mit_center_model = imread('center.jpg');
mit_center_model = imresize(mit_center_model,im_size);
mit_center_model = reshape(zscore(double(mit_center_model(:))),im_size);

% our gaussian center model
% center model
center_model = fspecial('gaussian',600,600);
center_model = imresize(center_model,im_size,'method','bilinear');
center_model = reshape(zscore(center_model(:)),im_size);
% rescale from 0 to 1
center_model_for_correction = (center_model(:)-min(center_model(:))) / (max(center_model(:))-min(center_model(:)));
center_model_for_correction = reshape(center_model_for_correction,im_size);

% define gaussian kernels
sigmas = [24 24 28];
for cost = 1:numel(cost_functions)
    gauss_kernels{cost} = fspecial('gaussian',6*sigmas(cost),sigmas(cost));
end

% loop over internal and external validation
for val_type = 1:numel(validation_types)
    fprintf('%s\n',validation_types{val_type});
    
    % Load fixation data
    fprintf('Loading Fixation Data\n');
    if val_type==1 % internal validation
        params.exp_path = exp_path;
        params.fixs_before = 2000; % use all fixations before (ms)
        for s = 1:numel(subs)
            [fix_inds(s,:),~,~,~] = load_fixation_data_salRecon(s,im_names,params);
        end
    else % external validation
        external_fix_dat = load(external_val_fix_file);
        fix_inds = external_fix_dat.fix_inds; clear external_fix_dat;
        fix_inds = fix_inds;
    end
    % make fixation maps (no smoothing)
    for s = 1:size(fix_inds,1)
        for im = 1:size(fix_inds,2)
            cur_map = zeros(im_size);
            if ~isempty(fix_inds{s,im})
                cur_map(fix_inds{s,im}) = 1;
            else
                cur_map = NaN(im_size);
            end
            fix_maps(s,im,:,:) = cur_map;
        end
    end
    fix_counts = cellfun(@numel,fix_inds);
    clear fix_inds;
    
    % load empirical baseline model
    fprintf('Loading Empirical Center-Bias Distributions\n');
    base_models = load(sprintf('%s/gold_standard_and_baseline_maps.mat',out_path),'baseline_maps');
    baseline_maps = base_models.baseline_maps{1}; clear base_models;
    % rescale from 0 to 1
    for im = 1:size(im_names)
        cur_map = squeeze(baseline_maps(im,:,:));
        cur_map = (cur_map(:) - min(cur_map(:))) / (max(cur_map(:))-min(cur_map(:)));
        baseline_maps(im,:,:) = reshape(cur_map,im_size);
    end

    % loop over CNN training regimes
    for cost = 1:numel(cost_functions)
        fprintf('CNN Training Set: %s\n',cost_functions{cost});

        % set path for current cost function
        feat_path = sprintf('%s/data/stimuli/vgg16_%s_activity',exp_path,cost_functions{cost});

        % calculate computational priority maps
        fprintf('Calculating Computational Priority Maps...');
        for im = 1:numel(im_names)
            fprintf('%d..',im);
            % load activity
            cur_activity = load(sprintf('%s/%s.mat',feat_path,im_names{im}));
            % loop over layers
            av_layer_act = NaN([numel(layers),im_size]);
            for layer = 1:numel(layers)
                layer_act = getfield(cur_activity,layers{layer}); %#ok<GFLD>
                % average across features
                layer_av = imresize(squeeze(mean(layer_act)),im_size,'method','bilinear'); % average, resize
                av_layer_act(layer,:,:) = reshape(zscore(layer_av(:)),size(layer_av)); % normalize
            end
            clear cur_activity;
            % average across features/channels
            mod1_base_nonorm = squeeze(mean(av_layer_act));
            mod1{1}(im,:,:) = reshape(zscore(mod1_base_nonorm(:)),im_size);
            % smoothing
            mod1_smoothed_nonorm = imfilter(mod1_base_nonorm,gauss_kernels{cost},'conv');
            mod1{2}(im,:,:) = reshape(zscore(mod1_smoothed_nonorm(:)),im_size);
            % center-bias correction
%             mod1_centered_nonorm = mod1_base_nonorm .* squeeze(baseline_maps(im,:,:));
            mod1_centered_nonorm = mod1_base_nonorm .* center_model_for_correction;
            mod1{3}(im,:,:) = reshape(zscore(mod1_centered_nonorm(:)),im_size);
            % smoothing + center-bias correction
            mod1_total_nonorm = imfilter(mod1_base_nonorm,gauss_kernels{cost},'conv');
%             mod1_total_nonorm = mod1_total_nonorm .* squeeze(baseline_maps(im,:,:));
            mod1_total_nonorm = mod1_total_nonorm .* center_model_for_correction;
            mod1{4}(im,:,:) = reshape(zscore(mod1_total_nonorm(:)),im_size);
            % visualize
            subplot(2,2,1);imagesc(squeeze(mod1{1}(im,:,:)),[-3 3]); axis off; title('Base','fontsize',20);
            subplot(2,2,2);imagesc(squeeze(mod1{2}(im,:,:)),[-3 3]); axis off; title('Smoothed','fontsize',20);
            subplot(2,2,3);imagesc(squeeze(mod1{3}(im,:,:)),[-3 3]); axis off; title('Centered','fontsize',20);
            subplot(2,2,4);imagesc(squeeze(mod1{4}(im,:,:)),[-3 3]); axis off; title('Total','fontsize',20);
            pause(.01);
        end
        % calculate NSS - spatial attention model
        fprintf('\nCalculating NSS\n');
        for mod = 1:numel(mod1)
            fprintf('%s\n',comp_model_types{mod});
            nss_calc = permute(repmat(mod1{mod},1,1,1,size(fix_maps,1)),[4 1 2 3]) .* fix_maps;
            nss_calc = reshape(nss_calc,size(fix_maps,1),numel(im_names),prod(im_size));
            fix_pred.mod_nss{val_type}(cost,mod,:,:) = sum(nss_calc,3)./fix_counts;
            clear nss_calc;
        end
        save(sprintf('%s/comp_model_priority_maps_gaussian_center_bias_correction_natcomm_v3_%s.mat',out_path,cost_functions{cost}),'mod1','-v7.3');
    end
    clear baseline_models;
    
    % benchmark models
    
    % mit center-bias
    nss_calc = permute(repmat(mit_center_model,1,1,size(fix_maps,2),size(fix_maps,1)),[4 3 1 2]) .* fix_maps;
    nss_calc = reshape(nss_calc,size(fix_maps,1),numel(im_names),prod(im_size));
    fix_pred.mit_center_nss{val_type} = sum(nss_calc,3)./fix_counts;
    clear nss_calc;
    
    % revision 1 center-bias
    nss_calc = permute(repmat(center_model,1,1,size(fix_maps,2),size(fix_maps,1)),[4 3 1 2]) .* fix_maps;
    nss_calc = reshape(nss_calc,size(fix_maps,1),numel(im_names),prod(im_size));
    fix_pred.r1_center_nss{val_type} = sum(nss_calc,3)./fix_counts;
    clear nss_calc;
    
    % empirical benchmarks
    empirical_models = load(sprintf('%s/gold_standard_and_baseline_maps.mat', out_path));
    
    % baseline prior (fixation distribution for all images except test 
    % image in the opposite validation set)
    nss_calc = permute(repmat(empirical_models.baseline_maps{val_type},1,1,1,size(fix_maps,1)),[4 1 2 3]) .* fix_maps;
    nss_calc = reshape(nss_calc,size(fix_maps,1),numel(im_names),prod(im_size));
    fix_pred.baseline_nss{val_type} = sum(nss_calc,3)./fix_counts;
    clear nss_calc;
    
    % gold-standard (fixation distribution for test image in the opposite
    % validation set)
    nss_calc = permute(repmat(empirical_models.gold_standard_maps{val_type},1,1,1,size(fix_maps,1)),[4 1 2 3]) .* fix_maps;
    nss_calc = reshape(nss_calc,size(fix_maps,1),numel(im_names),prod(im_size));
    fix_pred.gold_standard_nss{val_type} = sum(nss_calc,3)./fix_counts;
    clear nss_calc;
    
end

save(sprintf('%s/comp_model_results_gaussian_center_bias_correction_natcomm_v3.mat',out_path),'fix_pred','-v7.3');

% Figures

% [comp v bench] x [internal v external]
figure;

% computational models
comp_colormap = [199 234 229;
                 128 205 193;
                 53 151 143;
                 1 102 94];
for val_type = 1:2
    sp(val_type) = subplot(2,2,val_type); hold on;
    bar(squeeze(mean(nanmean(fix_pred.mod_nss{val_type},4),3)));
    colormap(sp(val_type),comp_colormap/255);
    set(gca,'xtick',1:3,'xticklabels',cost_functions,'fontsize',15);
    ylabel('NSS'); 
    if val_type==1
        legend(comp_model_types)
    end
    title(sprintf('Computational Models - %s',validation_types{val_type}),'fontsize',16);
    hold off;
end

% benchmark models
bench_colormap = [140 81 10;
                  191 129 45;
                  246 232 195];
for val_type = 1:2
    sp(val_type+2) = subplot(2,2,val_type+2); hold on;
    bar([mean(nanmean(fix_pred.mit_center_nss{val_type})),...
         mean(nanmean(fix_pred.r1_center_nss{val_type})),...
         mean(nanmean(fix_pred.gold_standard_nss{val_type}));...
         0 0 0]);
    colormap(sp(val_type+2),bench_colormap/255);
    axis([.5 1.5 0 3.05]);
    set(gca,'xtick',1,'xticklabels',[],'fontsize',15);
    ylabel('NSS');
    title(sprintf('Benchmark Models - %s',validation_types{val_type}),'fontsize',16);
    if val_type==1
        legend(bench_model_types);
    end
    hold off;
end

% summary for each cost function (consistent scale)
figure; count=1;
for val_type = 1:numel(validation_types)
    for cost = 1:numel(cost_functions)
        mod_nss_cur = squeeze(mean(nanmean(fix_pred.mod_nss{val_type}(cost,:,:,:),4),3));
        sp(count) = subplot(2,4,count); hold on;
        if cost==1 || cost==2
            bar([mean(nanmean(fix_pred.mit_center_nss{val_type})),mean(nanmean(fix_pred.r1_center_nss{val_type})),...
                mod_nss_cur(1),mod_nss_cur(2),...
                mod_nss_cur(3),mod_nss_cur(4),mean(nanmean(fix_pred.gold_standard_nss{val_type}));...
                0 0 0 0 0 0 0]);
            axis([.5 1.5 0 3.05]);
            colormap(sp(count),[bench_colormap(1:2,:);comp_colormap(1:2,:);...
                      comp_colormap(3:4,:);bench_colormap(3,:)]/255);
        else
            bar([mod_nss_cur,mean(nanmean(fix_pred.mit_center_nss{val_type})),mean(nanmean(fix_pred.r1_center_nss{val_type})),...
                 mean(nanmean(fix_pred.gold_standard_nss{val_type}));...
                 0 0 0 0 0 0 0]);
            axis([.5 1.5 0 3.05]);
            colormap(sp(count),[comp_colormap;bench_colormap]/255);
        end
        set(gca,'xtick',1:8,'xticklabels',[],'fontsize',15);
        ylabel('NSS');
        title(sprintf('%s - %s',validation_types{val_type},cost_functions{cost}),'fontsize',15);
        hold off;
        if count~=3
            count=count+1;
        else
            count = 5;
        end
    end
end
sp(7) = subplot(2,4,[4 8]); hold on;
colormap(sp(7),[bench_colormap;comp_colormap]/255);
bar([-1 -1 -1 -1 -1 -1 -1 -1;
    0 0 0 0 0 0 0 0]);
axis off; axis([0 .01 0 .01]);
legend([bench_model_types comp_model_types],'fontsize',15);
hold off;
    
    
