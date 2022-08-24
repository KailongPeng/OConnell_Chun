function results
%
% thomas oconnell

% params
subs = [2 3 5 6 9 11 13 14 15 16 18];
ROIs = {'V1','V2','V3','V4','LOC','PPA','FFA','OPA','RSC','IPS','FEF'};
layers = {'pool1','pool2','pool3','pool4','pool5'};
cost_functions = {'places365','ILSVRC','face'};
cnn_type_labs = {'Scene','Object','Face','Random'};
comp_model_types = {'Base Model','Smoothed','C-B Corrected','Smoothed & C-B Corrected'};
validation_types = {'wiS','internal','external'};
val_types = {'wiS','in','ex'};
average_models = {'base','total'};
layer_models = {'base','sm'};


% paths
cur_dir = pwd;
dir_ids = strfind(cur_dir,'/');
exp_path = cur_dir(1:dir_ids(end-1)-1);comp_mod_results_path = sprintf('%s/outputs/computational_model_files',exp_path);
fix_pred_results_path = sprintf('%s/outputs/fixation_prediction',exp_path);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load computational spatial attention model results %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Average Model

% goal-directed CNNs
mod_out = load(sprintf('%s/comp_model_results_gaussian_center_bias_correction_natcomm_v3.mat',comp_mod_results_path));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load all decoding model results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Fixation Map Reconstruction\n');
for roi = 1:numel(ROIs)
    fprintf('%s...',ROIs{roi});
    % load results
    load(sprintf('%s/fixation_prediction_results_fixMapRecon_%s.mat',fix_pred_results_path,ROIs{roi}));
    fix_pred_out = structfun(@(x) nanmean(x,2),fix_pred_out,'un',0);
    % within subject
    recon.fix.wiS(roi,:) = fix_pred_out.wiS_nss;
    % internal validation
    recon.fix.in(roi,:) = fix_pred_out.internal_nss;
    % external validation
    recon.fix.ex(roi,:) = fix_pred_out.external_nss;
    % calculate p-value from empirical null distribution
    for val = 1:numel(validation_types)
        null_distribution = eval(sprintf('sort(fix_pred_out.%s_nss_perm);',validation_types{val}));
        cur_nss = eval(sprintf('mean(recon.fix.%s(roi,:));',val_types{val}));
        diff_dist = null_distribution - cur_nss;
        if numel(find(diff_dist<0))==numel(diff_dist)
            recon.fix.p(roi,val) = 1/1001;
        elseif numel(find(diff_dist<0))==numel(diff_dist)-1
            recon.fix.p(roi,val) = 1/1000;
        else
            nss_ind = find(diff_dist>0); nss_ind=nss_ind(1);
            recon.fix.p(roi,val) = 1-nss_ind/numel(diff_dist);
        end
    end
end

fprintf('Model-Based Reconstruction - Average across CNN kernels - Gaussian Center Bias Correction\n');
for cost = 1
    fprintf('%s....',cost_functions{cost});
    for roi = 1:numel(ROIs)
        fprintf('%s...',ROIs{roi})
        % load results
        load(sprintf('%s/fixation_prediction_results_%s_%s.mat',fix_pred_results_path,cost_functions{cost},ROIs{roi}));
        fix_pred_out.base = structfun(@(x) nanmean(x,2),fix_pred_out.base,'un',0);
        fix_pred_out.total = structfun(@(x) nanmean(x,2),fix_pred_out.total,'un',0);
        % within subject
        recon.av.wiS.base(cost,roi,:) = fix_pred_out.base.wiS_nss;
        recon.av.wiS.total(cost,roi,:) = fix_pred_out.total.wiS_nss;
        % internal validation
        recon.av.in.base(cost,roi,:) = fix_pred_out.base.internal_nss;
        recon.av.in.total(cost,roi,:) = fix_pred_out.total.internal_nss;
        % external validation
        recon.av.ex.base(cost,roi,:) = fix_pred_out.base.external_nss;
        recon.av.ex.total(cost,roi,:) = fix_pred_out.total.external_nss;
        % calculate p-value from empirical null distributions
        for mod = 1:numel(average_models)
            for val = 1:numel(validation_types)
                null_distribution = eval(sprintf('sort(fix_pred_out.%s.%s_nss_perm);',...
                    average_models{mod},validation_types{val}));
                cur_nss = eval(sprintf('mean(recon.av.%s.%s(cost,roi,:));',val_types{val},average_models{mod}));
                diff_dist = null_distribution - cur_nss;
                if numel(find(diff_dist<0))==numel(diff_dist)
                    p(cost,roi,val,mod) = 1/1001;
                elseif numel(find(diff_dist<0))==numel(diff_dist)-1
                    p(cost,roi,val,mod) = 1/1000;
                else
                    nss_ind = find(diff_dist>0); nss_ind=nss_ind(1);
                    p(cost,roi,val,mod) = 1-nss_ind/numel(diff_dist);
                end
            end
        end
    end
    fprintf('\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Manuscript Fig. 1, fixation map recon and places365 model-based recons %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cur_colors = [7 191 0;106 0 191;0 141 191];

figure;

% Fixation Map Reconstruction, W/I Subject
subplot(3,3,1); hold on;
title('Fixation Map - W/I','fontsize',20);
bar(squeeze(mean(recon.fix.wiS,2)),'facecolor',cur_colors(2,:)/255);
errorbar(squeeze(mean(recon.fix.wiS,2)),...
         squeeze(sem(recon.fix.wiS,2)),...
         'linestyle','none','color','black');
axis([.5 11.5 -.07 .2]);
set(gca,'xtick',1:11,'xticklabels',ROIs);
for roi = 1:numel(ROIs)
    if recon.fix.p(roi,1)<0.001 % (cost,roi,val,mod)
        text(roi,.2,'***')
    elseif recon.fix.p(roi,1)<0.0045
        text(roi,.2,'**')
    elseif recon.fix.p(roi,1)<0.01
        text(roi,.2,'*')
    end
end

% Fixation Map Reconstruction, Internal
subplot(3,3,4); hold on;
title('Fixation Map - Internal','fontsize',20);
bar(squeeze(mean(recon.fix.in,2)),'facecolor',cur_colors(2,:)/255);
errorbar(squeeze(mean(recon.fix.in,2)),...
         squeeze(sem(recon.fix.in,2)),...
         'linestyle','none','color','black');
axis([.5 11.5 -.04 .4]);
set(gca,'xtick',1:11,'xticklabels',ROIs);
for roi = 1:numel(ROIs)
    if recon.fix.p(roi,2)<0.001 % (cost,roi,val,mod)
        text(roi,.4,'***')
    elseif recon.fix.p(roi,2)<0.0045
        text(roi,.4,'**')
    elseif recon.fix.p(roi,2)<0.01
        text(roi,.4,'*')
    end
end

% Fixation Map Reconstruction, External
subplot(3,3,7); hold on;
title('Fixation Map - External','fontsize',20);
bar(squeeze(mean(recon.fix.ex,2)),'facecolor',cur_colors(2,:)/255);
errorbar(squeeze(mean(recon.fix.ex,2)),...
         squeeze(sem(recon.fix.ex,2)),...
         'linestyle','none','color','black');
axis([.5 11.5 -.04 .4]);
set(gca,'xtick',1:11,'xticklabels',ROIs);
for roi = 1:numel(ROIs)
    if recon.fix.p(roi,3)<0.001 % (cost,roi,val,mod)
        text(roi,.4,'***')
    elseif recon.fix.p(roi,3)<0.0045
        text(roi,.4,'**')
    elseif recon.fix.p(roi,3)<0.01
        text(roi,.4,'*')
    end
end

% Model-based, Base, W/I Subject
subplot(3,3,2); hold on;
title('Base model - W/I','fontsize',20);
bar(squeeze(mean(recon.av.wiS.base(1,:,:),3)),'facecolor',cur_colors(3,:)/255);
errorbar(squeeze(mean(recon.av.wiS.base(1,:,:),3)),...
         squeeze(sem(recon.av.wiS.base(1,:,:),3)),...
         'linestyle','none','color','black');
axis([.5 11.5 -.07 .2]);
set(gca,'xtick',1:11,'xticklabels',ROIs);
for roi = 1:numel(ROIs)
    if p(1,roi,1,1)<0.001 % (cost,roi,val,mod)
        text(roi,.2,'***')
    elseif p(1,roi,1,1)<0.0045
        text(roi,.2,'**')
    elseif p(1,roi,1,1)<0.01
        text(roi,.2,'*')
    end
end

% Model-based, Base, Internal
subplot(3,3,5); hold on;
title('Base model - Internal','fontsize',20);
bar(squeeze(mean(recon.av.in.base(1,:,:),3)),'facecolor',cur_colors(3,:)/255);
errorbar(squeeze(mean(recon.av.in.base(1,:,:),3)),...
         squeeze(sem(recon.av.in.base(1,:,:),3)),...
         'linestyle','none','color','black');
axis([.5 11.5 -.04 .4]);
set(gca,'xtick',1:11,'xticklabels',ROIs);
for roi = 1:numel(ROIs)
    if p(1,roi,2,1)<0.001 % (cost,roi,val,mod)
        text(roi,.4,'***')
    elseif p(1,roi,2,1)<0.0045
        text(roi,.4,'**')
    elseif p(1,roi,2,1)<0.01
        text(roi,.4,'*')
    end
end

% Model-based, Base, External
subplot(3,3,8); hold on;
title('Base model - External','fontsize',20);
bar(squeeze(mean(recon.av.ex.base(1,:,:),3)),'facecolor',cur_colors(3,:)/255);
errorbar(squeeze(mean(recon.av.ex.base(1,:,:),3)),...
         squeeze(sem(recon.av.ex.base(1,:,:),3)),...
         'linestyle','none','color','black');
axis([.5 11.5 -.04 .4]);
set(gca,'xtick',1:11,'xticklabels',ROIs);
for roi = 1:numel(ROIs)
    if p(1,roi,3,1)<0.001 % (cost,roi,val,mod)
        text(roi,.4,'***')
    elseif p(1,roi,3,1)<0.0045
        text(roi,.4,'**')
    elseif p(1,roi,3,1)<0.01
        text(roi,.4,'*')
    end
end
        
% Model-based, Total, W/I Subject
subplot(3,3,3); hold on;
title('Total model - W/I','fontsize',20);
bar(squeeze(mean(recon.av.wiS.total(1,:,:),3)),'facecolor',cur_colors(1,:)/255);
errorbar(squeeze(mean(recon.av.wiS.total(1,:,:),3)),...
         squeeze(sem(recon.av.wiS.total(1,:,:),3)),...
         'linestyle','none','color','black');
axis([.5 11.5 -.07 .2]);
set(gca,'xtick',1:11,'xticklabels',ROIs);
for roi = 1:numel(ROIs)
    if p(1,roi,1,2)<0.001 % (cost,roi,val,mod)
        text(roi,.2,'***')
    elseif p(1,roi,1,2)<0.0045
        text(roi,.2,'**')
    elseif p(1,roi,1,2)<0.01
        text(roi,.2,'*')
    end
end

% Model-based, Total, Internal
subplot(3,3,6); hold on;
title('Total model - Internal','fontsize',20);
bar(squeeze(mean(recon.av.in.total(1,:,:),3)),'facecolor',cur_colors(1,:)/255);
errorbar(squeeze(mean(recon.av.in.total(1,:,:),3)),...
         squeeze(sem(recon.av.in.total(1,:,:),3)),...
         'linestyle','none','color','black');
axis([.5 11.5 -.04 .4]);
set(gca,'xtick',1:11,'xticklabels',ROIs);
for roi = 1:numel(ROIs)
    if p(1,roi,2,2)<0.001 % (cost,roi,val,mod)
        text(roi,.4,'***')
    elseif p(1,roi,2,2)<0.0045
        text(roi,.4,'**')
    elseif p(1,roi,2,2)<0.01
        text(roi,.4,'*')
    end
end

% Model-based, Total, External
subplot(3,3,9); hold on;
title('Total model - External','fontsize',20);
bar(squeeze(mean(recon.av.ex.total(1,:,:),3)),'facecolor',cur_colors(1,:)/255);
errorbar(squeeze(mean(recon.av.ex.total(1,:,:),3)),...
         squeeze(sem(recon.av.ex.total(1,:,:),3)),...
         'linestyle','none','color','black');
axis([.5 11.5 -.04 .4]);
set(gca,'xtick',1:11,'xticklabels',ROIs);
for roi = 1:numel(ROIs)
    if p(1,roi,3,2)<0.001 % (cost,roi,val,mod)
        text(roi,.4,'***')
    elseif p(1,roi,3,2)<0.0045
        text(roi,.4,'**')
    elseif p(1,roi,3,2)<0.01
        text(roi,.4,'*')
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Manuscript Fig. 3. spatial attention model - places365  %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% computational models
comp_colormap = [0 180 255;0 205 177;0 230 100;0 255 22];
bench_colormap = [140 81 10;191 129 45;246 232 195];
figure; count = 1;
for val_type = 1:2
    % Computational Model
    sp(val_type) = subplot(2,2,count); hold on;
    hb=bar([squeeze(mean(nanmean(mod_out.fix_pred.mod_nss{val_type}(1,:,:,:),4),3));[0 0 0 0]]);
    axis([.5 1.5 0 1.5]);
    colormap(sp(val_type),comp_colormap/255);
    set(gca,'xtick',[],'fontsize',15);
    pause(.01);
    for ib = 1:numel(hb)
        xData = hb(ib).XData(1)+hb(ib).XOffset;
        errorbar(xData,squeeze(mean(nanmean(mod_out.fix_pred.mod_nss{val_type}(1,ib,:,:),4),3)),...
                 squeeze(sem(nanmean(mod_out.fix_pred.mod_nss{val_type}(1,ib,:,:),4),3)),...
                 'linestyle','none','color','black');
    end
        
    title(sprintf('%s',validation_types{val_type+1}),'fontsize',20);
    if val_type==1
        ylabel('NSS');
    else
        set(gca,'ytick',[]);
    end
    hold off;
    % Computational Model w/ center-bias and gold standard benchmarks
    mod_nss_cur = squeeze(mean(nanmean(mod_out.fix_pred.mod_nss{val_type}(1,:,:,:),4),3));
    mod_sem_cur = squeeze(sem(nanmean(mod_out.fix_pred.mod_nss{val_type}(1,:,:,:),4),3));
    sp(count) = subplot(2,2,count+2); hold on;
    dat_vec = [mean(nanmean(mod_out.fix_pred.mit_center_nss{val_type})),mean(nanmean(mod_out.fix_pred.r1_center_nss{val_type})),...
        mod_nss_cur(1),mod_nss_cur(2),mod_nss_cur(3),mod_nss_cur(4),mean(nanmean(mod_out.fix_pred.gold_standard_nss{val_type}))];
    sem_vec = [sem(nanmean(mod_out.fix_pred.mit_center_nss{val_type})),sem(nanmean(mod_out.fix_pred.r1_center_nss{val_type})),...
        mod_sem_cur(1),mod_sem_cur(2),mod_sem_cur(3),mod_sem_cur(4),sem(nanmean(mod_out.fix_pred.gold_standard_nss{val_type}))];
    hb=bar([dat_vec;0 0 0 0 0 0 0]);
    colormap(sp(count),[bench_colormap(1:2,:);comp_colormap(1:2,:);...
              comp_colormap(3:4,:);bench_colormap(3,:)]/255);
    pause(.01);
    for ib = 1:numel(hb)
        %XData property is the tick labels/group centers; XOffset is the offset
        %of each distinct group
        xData = hb(ib).XData(1)+hb(ib).XOffset;
        errorbar(xData,dat_vec(ib),sem_vec(ib),'linestyle','none','color','black');
    end 
    axis([.5 1.5 0 3.15]); set(gca,'xtick',[],'fontsize',15);
    if val_type==1
        ylabel('NSS');
    else
        legend([comp_model_types {'MIT Center Model','Current Center Model','Gold Standard'}],'fontsize',10); set(gca,'ytick',[]);
        set(gca,'ytick',[]);
    end
    hold off; count=count+1;
end
