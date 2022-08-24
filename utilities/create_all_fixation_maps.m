% generate all fixation maps
%
% thomas oconnell
addpath(genpath('~/scripts/fileExchange_functions'));

% paths
cur_dir = pwd;
dir_ids = strfind(cur_dir,'/');
exp_path = cur_dir(1:dir_ids(end-1)-1);
stim_file = sprintf('%s/salRecon_file_lists_MRI.mat',exp_path);

% params
subs = [2 3 5 6 9 11 13 14 15 16 18];
params.exp_path = exp_path;
%params.num_fixs = 100;
params.fixs_before = 2000; % use all fixations before (ms)
im_size = [600 800];

% define gaussian kernls
sigmas = [15 33 20]; % sigma for gaussian filter (first is spatial error for eyetracking data, second is 1 degree of visual angle, third is cross-validated across datasets)
for sig = 1:numel(sigmas)
    gauss_kernels{sig} = fspecial('gaussian',6*sigmas(sig),sigmas(sig));
end

% load image lists
stim_list = load(stim_file);
fnames = uniquecell(stim_list.salRecon_lists.files(1,:,:));
im_names = cellfun(@(x) x(1:end-4),fnames,'Un',0);

% load fixation data (internal validation)
for s = 1:numel(subs)
    [fix_inds(s,:),~,~,~] = load_fixation_data_salRecon(s,im_names,params);
end

% calculate fixation maps
for s = 1:numel(subs)
    disp(s);
    % make fixation maps
    for im = 1:size(fix_inds,2)
        cur_map = zeros(im_size);
        if ~isempty(fix_inds{s,im})
            cur_map(fix_inds{s,im}) = 1;
        else
            cur_map = NaN(im_size);
        end
        for sig = 1:numel(sigmas)
            FDMs{s,sig}(im,:,:) = imfilter(cur_map,gauss_kernels{sig},'conv');
        end
    end
end

% % visualize
% for im = 1:numel(im_names)
%     for sig = 1:numel(sigmas)
%         subplot(2,2,sig);
%         imagesc(squeeze(FDMs{1,sig}(im,:,:))); axis off;
%     end
%     pause(.5);
% end

save(sprintf('%s/all_fixation_maps_171017.mat',exp_path),'FDMs','im_names','-v7.3');
