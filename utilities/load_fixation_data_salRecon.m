function [fix_inds,durations_tar,onsets_tar,target_list,FDMs]=load_fixation_data_salRecon(s,target_list,params)

    % params
    subs = [2 3 5 6 9 11 13 14 15 16 18];   
    fix_dir = sprintf('%s/data/sub%d/eyetracking',params.exp_path,subs(s));
    im_res = [600 800];
    screen_res = [768 1024];
%    gauss_kernel = fspecial('gaussian',2*ceil(2*params.sigma)+1,params.sigma);
    runs=12; trials=24;
    
    % extract fixation data file names
    files = dir(sprintf('%s/fixation_report_sub%d_block*.txt',fix_dir,subs(s)));
    
    count = 1;
    filenames = {};
%     fprintf('Loading Fixation Data');
    for b = 1:runs
        % load fixation data
        x = tdfread(sprintf('%s/fixation_report_sub%d_block%d.txt',fix_dir,subs(s),b));
        % load condition data
        load(sprintf('%s/sub%d_block%d_conditions.mat',fix_dir,subs(s),b));
        filenames = [filenames out.filenames'];
        % parse data
        for trial = 1:trials
%             % load image
            %ims{count} = imread(sprintf('%s/%s',params.im_path,filenames{count}));
            % extract variables for current trial
            x_coords{count} = x.CURRENT_FIX_X(x.TRIAL_INDEX==trial);
            y_coords{count} = x.CURRENT_FIX_Y(x.TRIAL_INDEX==trial);
            durations{count} = x.CURRENT_FIX_DURATION(x.TRIAL_INDEX==trial);
            onsets{count} = x.CURRENT_FIX_START(x.TRIAL_INDEX==trial);
            % correct for screen resolution
            x_coords{count} = x_coords{count} - (screen_res(2)-im_res(2))/2;
            y_coords{count} = y_coords{count} - (screen_res(1)-im_res(1))/2;
            % round
            x_coords{count} = round(x_coords{count});
            y_coords{count} = round(y_coords{count});
            % remove fixations off screen
            for fix = 1:numel(x_coords{count})
                if x_coords{count}(fix)>im_res(2) || x_coords{count}(fix)<1 || y_coords{count}(fix)>im_res(1) || y_coords{count}(fix)<1
                    x_coords{count}(fix) = NaN;
                    y_coords{count}(fix) = NaN;
                    durations{count}(fix) = NaN;
                    onsets{count}(fix) = NaN;
                end
            end
            x_coords{count}(isnan(x_coords{count})) = [];
            y_coords{count}(isnan(y_coords{count})) = [];
            durations{count}(isnan(durations{count})) = [];
            onsets{count}(isnan(onsets{count})) = [];
            % only retain fixations from 200ms - 2000ms
            x_coords{count} = x_coords{count}(intersect(find(onsets{count}>200),find(onsets{count}<params.fixs_before)));
            y_coords{count} = y_coords{count}(intersect(find(onsets{count}>200),find(onsets{count}<params.fixs_before)));
            durations{count} = durations{count}(intersect(find(onsets{count}>200),find(onsets{count}<params.fixs_before)));
            onsets{count} = onsets{count}(intersect(find(onsets{count}>200),find(onsets{count}<params.fixs_before)));
            % exclude fixations by #
%             if numel(x_coords{count}(2:end))>=params.num_fixs
%                 x_coords{count} = x_coords{count}(2:params.num_fixs+1);
%                 durations{count} = durations{count}(2:params.num_fixs+1);
%             else
%                 x_coords{count} = x_coords{count}(2:end);
%             end
%             if numel(y_coords{count}(2:end))>=params.num_fixs
%                 y_coords{count} = y_coords{count}(2:params.num_fixs+1);
%             else
%                 y_coords{count} = y_coords{count}(2:end);
%             end
            count=count+1;  
        end
    end
    
    % extract fixations for target images
    if ischar(target_list)
        cur_ind = find(~cellfun(@isempty,strfind(filenames,target_list)));
        fix_inds = sub2ind(im_res,y_coords{cur_ind},x_coords{cur_ind});
        durations_tar = durations{cur_ind};
        onsets_tar = onsets{cur_ind};
        %images = ims{cur_ind};
    else
        for im = 1:numel(target_list)
            cur_ind = find(~cellfun(@isempty,strfind(filenames,target_list{im})));
            fix_inds{im} = sub2ind(im_res,y_coords{cur_ind},x_coords{cur_ind});
            durations_tar{im} = durations{cur_ind};
            onsets_tar{im} = onsets{cur_ind};
            %images{im} = ims{cur_ind};
        end
    end
    
%     % calculate duration-weighted fixation density maps
%    fprintf('\nCalculating Fixation Density Maps....');
%    FDMs = zeros(numel(target_list),im_res(1),im_res(2));
%    if ischar(target_list)
%        cur_FDM = zeros(im_res); cur_FDM(fix_inds) = durations_tar;
%        FDMs = imfilter(cur_FDM,gauss_kernel,'conv');
% %        FDMs = imgaussfilt(cur_FDM,params.sigma);
%    else
%        for im = 1:numel(target_list)
%            fprintf('%d..',im);
%            cur_FDM = zeros(im_res); cur_FDM(fix_inds{im}) = durations_tar{im};
%            FDMs(im,:,:) = imfilter(cur_FDM,gauss_kernel,'conv');
% %            FDMs(im,:,:) = imgaussfilt(cur_FDM,params.sigma);
%        end
%    end

end                
