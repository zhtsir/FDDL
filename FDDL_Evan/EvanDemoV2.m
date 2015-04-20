close all;
clear all;
clc;

addpath('F:\Graduate Design\Database');
load('MCI_data.mat');

% ind1 = find(sum(pMCI_data,2) == 0);
% ind2 = find(sum(sMCI_data,2) == 0);
% c = intersect(ind1,ind2);
% c1 = setdiff(ind1,c);
% c2 = setdiff(ind2,c);
% ind = [c;c1;c2];

% pMCI_data(ind,:) = [];
% sMCI_data(ind,:) = [];

data = [pMCI_data,sMCI_data];
data_label = [1*ones(1,size(pMCI_data,2)),2*ones(1,size(sMCI_data,2))];
data = double(data);
ind = find(sum(data,2) == 0);
data(ind,:) = [];
clear pMCI_data sMCI_data ind ind1 ind2

%% Feature Selection by PCA


[COEFF,SCORE,latent] = princomp(zscore(data'),'econ');
% calculate the contribution rate, larger than 88%
CumVal = cumsum(latent)./sum(latent);
EigenNum = find(CumVal >= 0.88);
EigenNum = EigenNum(1);
% EigenNumAll(j+1) = EigenNum;

data = (data'*COEFF(:,1:EigenNum))';

% EigenNum = 200;
% P = [];
% [row,col] = size(data);
% MeanData = data - repmat(mean(data,2),1,col);
% C = 1/col.*MeanData*MeanData';
% [V,S] = eig(C);
% S = S*ones(1,size(S,1));
% [Y,Index] = sort(S,'descend');
% for i = 1:EigenNum
%     P = [P;V(:,Index(i))'];
% end
% data = P*data;    

c = cvpartition(data_label,'k',5);
save('cvpartition_all_5','c');
% load cvpartition_all.mat;

for k = 1:5

    Xt = data(:,training(c,k));
    Lt = data_label(:,training(c,k));
    Xs = data(:,test(c,k));
    Ls = data_label(:,test(c,k));

    Xt = Xt./repmat(sqrt(sum(Xt.^2)),size(Xt,1),1);
    Xs = Xs./repmat(sqrt(sum(Xs.^2)),size(Xs,1),1);

    lambda = 0.005;
%     [ACC(k),LABEL,C] = SLEP_LeastR_SparseClassify_Evan1(Xt,Lt,Xs,Ls,lambda);

    %% FDDL Parameter

    opts.nClass = 2;
    opts.wayInit = 'random';
    opts.dictnums = 200 ;%set the numbers of dictionary atom of each class(edit by Evan)
    opts.lambda1 = 0.005;
    opts.lambda2 = 0.1;%0.05;
    opts.nIter = 15;
    opts.show = true;
    [Dict,Drls,CoefM,CMlabel] = FDDL(Xt,Lt,opts);
%     filename = strcat('GMNewDict',num2str(k));
%     save(filename, 'Dict','Drls','CoefM','CMlabel');

    [ACC(k),LABEL,C] = SLEP_treeLeastR_SparseClassify_Evan1(Dict,Drls,Xs,Ls,lambda);
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    % Sparse Classification
    %%%%%%%%%%%%%%%%%%%%%%%%
%     lambda   =   0.005;
%     nClass   =   opts.nClass;
%     weight   =   0.5;
% 
%     td1_ipts.D    =   Dict;
%     td1_ipts.tau1 =   lambda;
%     if size(td1_ipts.D,1)>=size(td1_ipts.D,2)
%        td1_par.eigenv = eigs(td1_ipts.D'*td1_ipts.D,1);
%     else
%        td1_par.eigenv = eigs(td1_ipts.D*td1_ipts.D',1);  
%     end
% 
%     ID   =   [];
%     for indTest = 1:size(Xs,2)
%         fprintf(['Totalnum:' num2str(size(Xs,2)) 'Nowprocess:' num2str(indTest) '\n']);
%         td1_ipts.y          =      Xs(:,indTest);   
%         [opts]              =      IPM_SC(td1_ipts,td1_par);
%         s                   =      opts.x;
% 
%         for indClass  =  1:nClass
%             temp_s            =  zeros(size(s));
%             temp_s(indClass==Drls) = s(indClass==Drls);
%             zz                =  Xs(:,indTest)-td1_ipts.D*temp_s;
%             gap(indClass)     =  zz(:)'*zz(:);
% 
%             mean_coef_c         =   CoefM(:,indClass);
%             gCoef3(indClass)    =  norm(s-mean_coef_c,2)^2;    
%         end
% 
%         wgap3  = gap + weight*gCoef3;
%         index3 = find(wgap3==min(wgap3));
%         id3    = index3(1);
%         ID     = [ID id3];
%     end  
% 
%     fprintf('%s%8f\n','reco_rate  =  ',sum(ID==Ls)/(length(Ls)));
end


