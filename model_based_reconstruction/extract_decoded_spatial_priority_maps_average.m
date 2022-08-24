function extract_decoded_spatial_priority_maps_average(index)
% computer model-based reconstructed spatial priority maps
% 16 GB memory load
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
%cost_functions = {'places365','ILSVRC','face'};
cost_functions = {'places365'};
im_size = [600 800];
map_size = [112 112];

[A,B] = meshgrid([1:numel(cost_functions)],[1:numel(ROIs)]);
c=cat(2,A',B');
d=reshape(c,[],2);
cost = d(index,1)
roi = d(index,2)

% paths
cur_dir = pwd;
dir_ids = strfind(cur_dir,'/');
exp_path = cur_dir(1:dir_ids(end-1)-1);
stim_file = sprintf('%s/data/salRecon_file_lists_MRI.mat',exp_path);
recon_path = sprintf('%s/outputs/model_aligned_bold_activity',exp_path);
out_path = sprintf('%s/outputs/reconstructions',exp_path);
addpath(genpath(sprintf('%s/scripts/utilities',exp_path)));

% load image lists
stim_list = load(stim_file);
fnames = unique(stim_list.salRecon_lists.files(1,:,:));
im_names = cellfun(@(x) x(1:end-4),fnames,'Un',0);

% center model (matched to MIT Benchmark Center Model)
center_model = fspecial('gaussian',600,600);
center_model = imresize(center_model,im_size,'method','bilinear');
center_model = (center_model(:) - min(center_model(:))) / (max(center_model(:))-min(center_model(:)));
center_model = reshape(center_model,im_size);

% create gaussian kernel
% sigmas = [15 33]; %1: spatial error of eye tracking data, 2: 1 degree of visual angle
sigmas = [24 24 28]; % optimized smoothing kernels for internal validation
for sig = 1:numel(sigmas)
    gauss_kernels{sig} = fspecial('gaussian',2*ceil(2*sigmas(sig))+1,sigmas(sig));
end

fprintf('Generating Model-Based Reconstructed Priority Maps - %s\n',ROIs{roi});
fprintf('CNN Training Set: %s\n',cost_functions{cost});
% declare arrays
recons_average_base = NaN(numel(subs),numel(im_names),im_size(1),im_size(2));
recons_average_total = NaN(numel(subs),numel(im_names),im_size(1),im_size(2));
for s = 1:numel(subs)
    fprintf('Sub%d....',subs(s));
    % declare array
    recons_all_layers = NaN(numel(layers),numel(im_names),im_size(1),im_size(2));
    memcheck;
    for layer = 1:numel(layers)
        fprintf('%s...',layers{layer});
        % load data
        all_cur_dat = load(sprintf('%s/sub%d_%s_vgg_%s_%s_aligned_bold.mat',...
            recon_path,subs(s),ROIs{roi},cost_functions{cost},layers{layer}));
        snames = all_cur_dat.sub_file_list;
        snames = cellfun(@(x) x(1:end-4),snames,'Un',0);
        % reshape
        cur_recon = reshape(all_cur_dat.model_aligned_bold,...
            [numel(all_cur_dat.sub_file_list) layer_dims(layer,:)]);
        % average across repetitions
        for im = 1:numel(im_names)
            cur_inds = find(strcmp(snames,im_names{im}));
            if ~isempty(cur_inds)
                cur_recon_repAv(im,:,:,:) = squeeze(mean(cur_recon(cur_inds,:,:,:)));
            else
                cur_recon_repAv(im,:,:,:) = NaN(layer_dims(layer,:));
            end
        end
        clear cur_recon;
        % Average across features
        recon_average = squeeze(mean(cur_recon_repAv,2));
        clear cur_recon_repAv;
        % Resize, smooth, normalize
        for im = 1:size(recon_average,1)
            recon_average_rs = imresize(squeeze(recon_average(im,:,:)),im_size,'method','bilinear');
            recons_all_layers(layer,im,:,:) = reshape(zscore(recon_average_rs(:)),im_size);
        end
        memcheck;
        clear recon_average recon_average_rs recon_average_sm;
    end
    % Average across layers
    for im = 1:numel(im_names)
        fprintf('%d..',im);
        % Base
        recon_base = squeeze(mean(recons_all_layers(:,im,:,:)));
        recons_average_base(s,im,:,:) = reshape(zscore(recon_base(:)),im_size);
        % Total (Smoothing + Center-Bias Correction)
        recon_total = imfilter(recon_base,gauss_kernels{cost},'conv');
%         recon_total = recon_total .* squeeze(baseline_maps(im,:,:));
        recon_total = recon_total .* center_model;
        recons_average_total(s,im,:,:) = reshape(zscore(recon_total(:)),im_size);
    end
    clear recons_all_layers recons_all_layers_sm;
    fprintf('\n');
end
save(sprintf('%s/model_based_reconstructions_%s_%s.mat',out_path,cost_functions{cost},ROIs{roi}),...
    'recons_average_base','recons_average_total','im_names','-v7.3');
clear recons_average_base recons_average_total;
