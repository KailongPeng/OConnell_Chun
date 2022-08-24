function extract_decoded_fixation_maps(roi)
% extract directly reconstructed fixation maps
% 25 GB memory load
%
% thomas oconnell

% params
subs = [2 3 5 6 9 11 13 14 15 16 18];
ROIs = {'V1','V2','V3','V4','LOC','PPA','FFA','OPA','RSC','IPS','FEF'};
im_size = [600 800];


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

% declare array
recons_fix_map_all_subs = NaN(numel(subs),numel(im_names),im_size(1),im_size(2));

for s = 1:numel(subs)
    fprintf('Sub%d....',subs(s));
    % load data
    all_cur_dat = load(sprintf('%s/sub%d_%s_fixation_map_aligned_bold.mat',...
        recon_path,subs(s),ROIs{roi}));
    snames = all_cur_dat.sub_file_list;
    snames = cellfun(@(x) x(1:end-4),snames,'Un',0);
    % average across repetitions
    for im = 1:numel(im_names)
        cur_inds = find(strcmp(snames,im_names{im}));
        if ~isempty(cur_inds)
            cur_map = squeeze(mean(all_cur_dat.fix_map_recons(cur_inds,:,:)));
            recons_fix_map_all_subs(s,im,:,:) = reshape(zscore(cur_map(:)),size(cur_map));
        else
            recons_fix_map_all_subs(s,im,:,:) = NaN(im_size);
        end
    end
    clear all_cur_dat snames;
end
% save fixation map reconstructions
save(sprintf('%s/fixation_map_reconstructions_%s.mat',out_path,ROIs{roi}),'recons_fix_map_all_subs','im_names','-v7.3');
