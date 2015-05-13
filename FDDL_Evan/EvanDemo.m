close all;
clear all;
clc;

%% Dispose Data of MCI
addpath('F:\Graduate Design\Database');
addpath('F:\Graduate Design\Database\ROI');
load('MCI403_ROI_5tpt');

[row,col,cell] = size(pMCI_data);
pMCI_1 = reshape(pMCI_data(:,1,:),[row,cell]);
pMCI_2 = reshape(pMCI_data(:,2,:),[row,cell]);
% pMCI_3 = reshape(pMCI_data(:,3,:),[row,cell]);
% pMCI_4 = reshape(pMCI_data(:,4,:),[row,cell]);
% pMCI_5 = reshape(pMCI_data(:,5,:),[row,cell]);

[row,col,cell] = size(sMCI_data);
sMCI_1 = reshape(sMCI_data(:,1,:),[row,cell]);
sMCI_2 = reshape(sMCI_data(:,2,:),[row,cell]);
% sMCI_3 = reshape(sMCI_data(:,3,:),[row,cell]);
% sMCI_4 = reshape(sMCI_data(:,4,:),[row,cell]);
% sMCI_5 = reshape(sMCI_data(:,5,:),[row,cell]);

datalabel = [ones(1,size(pMCI_1,2)),2.*ones(1,size(sMCI_1,2))];
data_1 = [pMCI_1,sMCI_1];
data_1 = data_1(1:4:size(data_1,1),:);
data_2 = [pMCI_2,sMCI_2];
data_2 = data_2(1:4:size(data_2,1),:);
% data_3 = [pMCI_3,sMCI_3];
% data_3 = data_3(1:4:size(data_3,1),:);
% data_4 = [pMCI_4,sMCI_4];
% data_4 = data_4(1:4:size(data_4,1),:);
% data_5 = [pMCI_5,sMCI_5];
% data_5 = data_5(1:4:size(data_5,1),:);

%% Dispose Data of AD/NORMAL
% addpath('F:\Graduate Design\Database');
% addpath('F:\Graduate Design\Database\ROI');
% load('AD198_ROI_5tpt.mat');
% load('NORMAL229_ROI_5tpt.mat');
% 
% [row,col,cell] = size(AD_data);
% AD = reshape(AD_data,[row,cell,col]);
% [row,col,cell] = size(NORMAL_data);
% NORMAL = reshape(NORMAL_data,[row,cell,col]);
% AD_1 = AD(:,:,1);NORMAL_1 = NORMAL(:,:,1);
% datalabel = [ones(1,size(AD_1,2)),2.*ones(1,size(NORMAL_1,2))];
% data = [AD_1,NORMAL_1];
% ind = find(sum(data,1) ~= 0);
% for i = 1:size(data,1)
%     if any(ind == i)
%     data(:,i) = data(:,i)./repmat(sqrt(sum(data(:,i).^2)),size(data,1),1);
%     end
% end

% ind = find(sum(data,1) == 0);
% data(:,ind) = [];
% datalabel(:,ind) = [];
% data = data./repmat(sqrt(sum(data.^2)),size(data,1),1);

clear  row col cell 
clear  pMCI_1 pMCI_2 pMCI_3 pMCI_4 pMCI_5;
clear  sMCI_1 sMCI_2 sMCI_3 sMCI_4 sMCI_5;

% c = cvpartition(datalabel,'k',10);
% save('cvpartition','c');
load Cpartition;
LAMBDA = [1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005];
for j = 7:7
    for k = 1:10
        
%         Xt = data_1(:,training(c,k));
%         Lt = datalabel(:,training(c,k));
        
        Xt_1 = data_1(:,training(c,k));
        Lt_1 = datalabel(:,training(c,k));
        Xt_2 = data_2(:,training(c,k));
        Lt_2 = datalabel(:,training(c,k));
%         Xt_3 = data_3(:,training(c,k));
%         Lt_3 = datalabel(:,training(c,k));
%         Xt_4 = data_4(:,training(c,k));
%         Lt_4 = datalabel(:,training(c,k));
%         Xt_5 = data_5(:,training(c,k));
%         Lt_5 = datalabel(:,training(c,k));
        
        Xs_1 = data_1(:,test(c,k));
        Ls_1 = datalabel(:,test(c,k));
%         Xs_2 = data_2(:,test(c,k));
%         Ls_2 = datalabel(:,test(c,k));
%         Xs_3 = data_3(:,test(c,k));
%         Ls_3 = datalabel(:,test(c,k));
%         Xs_4 = data_4(:,test(c,k));
%         Ls_4 = datalabel(:,test(c,k));
%         Xs_5 = data_5(:,test(c,k));
%         Ls_5 = datalabel(:,test(c,k));

        flag = find(sum(Xt_2,1) == 0);
        Xt_2(:,flag) = Xt_1(:,flag);
%         flag1 = find(sum(Xt_3,1) == 0);
%         Xt_3(:,flag1) = Xt_1(:,flag1);
%         flag = find(sum(Xt_4,1) == 0);
%         Xt_4(:,flag) = Xt_3(:,flag);

        
%         Xt = zeros(size(Xt_1,1),5*size(Xt_1,2));
%         Xt(:,1:5:5*(size(Xt_1,2)-1)+1) = Xt_1;
%         Xt(:,2:5:5*(size(Xt_1,2)-1)+2) = Xt_2;
%         Xt(:,3:5:5*(size(Xt_1,2)-1)+3) = Xt_3;
%         Xt(:,4:5:5*(size(Xt_1,2)-1)+4) = Xt_4;
%         Xt(:,5:5:5*(size(Xt_1,2))) = Xt_5;
%         
%         Xs = zeros(size(Xs_1,1),5*size(Xs_1,2));
%         Xs(:,1:5:5*(size(Xs_1,2)-1)+1) = Xs_1;
%         Xs(:,2:5:5*(size(Xs_1,2)-1)+2) = Xs_2;
%         Xs(:,3:5:5*(size(Xs_1,2)-1)+3) = Xs_3;
%         Xs(:,4:5:5*(size(Xs_1,2)-1)+4) = Xs_4;
%         Xs(:,5:5:5*(size(Xs_1,2))) = Xs_5;
%         
%         Lt = zeros(size(Lt_1,1),5*size(Lt_1,2));
%         Lt(:,1:5:5*(size(Lt_1,2)-1)+1) = Lt_1;
%         Lt(:,2:5:5*(size(Lt_1,2)-1)+2) = Lt_2;
%         Lt(:,3:5:5*(size(Lt_1,2)-1)+3) = Lt_3;
%         Lt(:,4:5:5*(size(Lt_1,2)-1)+4) = Lt_4;
%         Lt(:,5:5:5*(size(Lt_1,2))) = Lt_5;
%         
%         Ls = zeros(size(Ls_1,1),5*size(Ls_1,2));
%         Ls(:,1:5:5*(size(Ls_1,2)-1)+1) = Ls_1;
%         Ls(:,2:5:5*(size(Ls_1,2)-1)+2) = Ls_2;
%         Ls(:,3:5:5*(size(Ls_1,2)-1)+3) = Ls_3;
%         Ls(:,4:5:5*(size(Ls_1,2)-1)+4) = Ls_4;
%         Ls(:,5:5:5*(size(Ls_1,2))) = Ls_5;
        
        
%         id = [1:1:size(Xs_1,2)];
%         id = kron(id,[1,1,1,1,1]);
%         id_label = Ls_1;
        
%         ind = find(sum(Xs,1) == 0);
%         ind2 = find(sum(Xt,1) == 0);
%         Xs(:,ind) = [];
%         Xt(:,ind2) = [];
%         Ls(:,ind) = [];
%         Lt(:,ind2) = [];
%         id(:,ind) = [];

        %% Normalization
        Xt_1 = Xt_1./repmat(sqrt(sum(Xt_1.^2)),size(Xt_1,1),1);
        Xt_2 = Xt_2./repmat(sqrt(sum(Xt_2.^2)),size(Xt_2,1),1);
%         Xt_3 = Xt_3./repmat(sqrt(sum(Xt_3.^2)),size(Xt_3,1),1);
%         Xt_4 = Xt_4./repmat(sqrt(sum(Xt_4.^2)),size(Xt_4,1),1);
        
        Xs_1 = Xs_1./repmat(sqrt(sum(Xs_1.^2)),size(Xs_1,1),1);
        

%% FDDL Parameter
        
% %         opts.nClass = 2;
% %         opts.wayInit = 'pca';
% %         opts.dictnums = 93 ;%set the numbers of dictionary atom of each class(edit by Evan)
% %         opts.lambda1 = 0.005;
% %         opts.lambda2 = 0.05;
% %         opts.nIter = 12;
% %         opts.show = true;
% % 
% % %         [Dict1,Drls1,CoefM1,CMlabel1] = FDDL(Xt_1,Lt_1,opts);
% %         
% % %         [Dict2,Drls2,CoefM2,CMlabel2] = FDDL(Xt_2,Lt_2,opts);
% %         
% % %         [Dict3,Drls3,CoefM3,CMlabel3] = FDDL(Xt_3,Lt_3,opts);
% %         
% % %         [Dict4,Drls4,CoefM4,CMlabel4] = FDDL(Xt_4,Lt_4,opts);
% % 
% %         filename = strcat('Dict051102_',num2str(k));
% %         save(filename, 'Dict3','Drls3','CoefM3','CMlabel3');
     
 %% Sparse Classification I
% %         filename = strcat('Dict050901_',num2str(k));
% %         load(filename);
% %         opts.nClass = 2;
% %         lambda   =   0.005;
% %         nClass   =   opts.nClass;
% %         weight   =   0.05;
    
% %         td1_ipts.D    =   Dict1;
% %         td1_ipts.tau1 =   lambda;
% %         if size(td1_ipts.D,1)>=size(td1_ipts.D,2)
% %            td1_par.eigenv = eigs(td1_ipts.D'*td1_ipts.D,1);
% %         else
% %            td1_par.eigenv = eigs(td1_ipts.D*td1_ipts.D',1);  
% %         end
    
        
% %         ID   =   [];
% %         gap = [];
% %         gCoef3 = [];
% %         for indTest = 1:size(Xt_1,2)
% %             
% %             fprintf(['Totalnum:' num2str(size(Xt_1,2)) 'Nowprocess:' num2str(indTest) '\n']);
% %             td1_ipts.y          =      Xt_1(:,indTest);   
% %             [opts]              =      IPM_SC(td1_ipts,td1_par);
% %             s                   =      opts.x;
% %     
% %             for indClass  =  1:nClass
% %                 temp_s            =  zeros(size(s));
% %                 temp_s(indClass==Drls1) = s(indClass==Drls1);
% %                 zz                =  Xt_1(:,indTest)-td1_ipts.D*temp_s;
% %                 gap(indTest,indClass)     =  zz(:)'*zz(:);
% %                     
% %                 mean_coef_c         =   CoefM1(:,indClass);
% %                 gCoef3(indTest,indClass)    =  norm(s-mean_coef_c,2)^2;
% %                 
% % % %                 filename = strcat('0512GapFold',num2str(k));
% % % %                 save(filename,'gap','gCoef3')
% %             end
% %             filename = strcat('0512GapFold',num2str(k));
% %             load(filename)
% % 
% %             wgap3  =  gap + weight * gCoef3;
% %             index3 = find(wgap3(indTest,:)==min(wgap3(indTest,:)));
% %             id3    = index3(1);
% %             ID     = [ID id3];
% %         end

% %         fprintf('%s%8f\n','reco_rate  =  ',sum(ID==Lt_1)/(length(Lt_1)));
% %         ACC(k) = sum(ID==Lt_1)/(length(Lt_1));        

%% Calculate the Const parameters        

% % 
% %         filename = strcat('0512GapFold',num2str(k));
% %         load(filename);
% %         Const = 0;
% %         step = 0.005;
% %         is_loop = 1;
% %         StopCri = 0.000001;
% %         nIter = 100;
% %         iter = 0;
% %         direction = +1;
% %         Lt_1_1 = Lt_1((Lt_1 == 1));
% %         Lt_1_2 = Lt_1((Lt_1 == 2));
% %         
% %         cri = gap(:,1) - gap(:,2) - Const;
% %         ID = sign(cri)/2 + 3/2;
% %         ID = ID';
% %         ID_1 = ID(Lt_1 == 1);
% %         ID_2 = ID(Lt_1 == 2);
% % 
% %         
% % %         fprintf('%s%8f\n','reco_rate  =  ',sum(ID==Lt_1)/(length(Lt_1)));
% %         ACC_c1 = sum(ID_1 == Lt_1_1)/length(Lt_1_1);
% %         ACC_c2 = sum(ID_2 == Lt_1_2)/length(Lt_1_2);
% %         ACC_all = sum(ID==Lt_1)/(length(Lt_1));
% %         ACCmax = ACC_all;
% %        
% %         
% %         while(is_loop || iter <= nIter)
% %             
% %             Const = Const + direction * step;
% %             cri = gap(:,1) - gap(:,2) - Const;
% %             ID = sign(cri)/2 + 3/2;
% %             ID = ID';
% %             ID_1 = ID(Lt_1 == 1);
% %             ID_2 = ID(Lt_1 == 2);            
% %             ACC_temp = sum(ID == Lt_1)/length(Lt_1)
% %             ACC_c1_temp = sum(ID_1 == Lt_1_1)/length(Lt_1_1);
% %             ACC_c2_temp = sum(ID_2 == Lt_1_2)/length(Lt_1_2);
% %             
% %             if abs(ACC_temp - ACCmax) < StopCri
% %                 is_loop = 0;
% %             end
% %             
% %             if ACC_temp > ACCmax
% %                 ACCmax = ACC_temp;
% %                 ACC(k) = ACCmax;
% %                 CONST(k) = Const;
% %             end
% %             
% %             if ACC_temp > ACC_all
% %                 ACC_all = ACC_temp;
% %                 ACC_c1 = ACC_c1_temp;
% %                 ACC_c2 = ACC_c2_temp;
% % 
% %                 continue;
% %             elseif ACC_c1_temp <= ACC_c1 && ACC_c2_temp >= ACC_c2
% %                 direction = 1;
% %                 step = step * 0.9;
% %                 
% %                 ACC_all = ACC_temp;
% %                 ACC_c1 = ACC_c1_temp;
% %                 ACC_c2 = ACC_c2_temp;
% %             elseif ACC_c1_temp >= ACC_c1 && ACC_c2_temp <= ACC_c2
% %                 direction = -1;
% %                 step = step * 0.9;
% %                 
% %                 ACC_all = ACC_temp;
% %                 ACC_c1 = ACC_c1_temp;
% %                 ACC_c2 = ACC_c2_temp;
% %                 
% %             end
% % 
% %             iter = iter + 1;           
% %         end
% %         save('CONST','CONST')

%% Classification       
% %         ID   =   [];
% %         gap = [];
% %         gCoef3 = [];
% %         for indTest = 1:size(Xs_1,2)
% %             fprintf(['Totalnum:' num2str(size(Xs_1,2)) 'Nowprocess:' num2str(indTest) '\n']);
% %             td1_ipts.y          =      Xs_1(:,indTest);   
% %             [opts]              =      IPM_SC(td1_ipts,td1_par);
% %             s                   =      opts.x;
% %     
% %             for indClass  =  1:nClass
% %                 temp_s            =  zeros(size(s));
% %                 temp_s(indClass==Drls1) = s(indClass==Drls1);
% %                 zz                =  Xs_1(:,indTest)-td1_ipts.D*temp_s;
% %                 gap(indTest,indClass)     =  zz(:)'*zz(:);
% %                     
% %                 mean_coef_c         =   CoefM1(:,indClass);
% %                 gCoef3(indTest,indClass)    =  norm(s-mean_coef_c,2)^2;
% %                 
% %                 filename = strcat('test0512GapFold',num2str(k));
% %                 save(filename,'gap','gCoef3')
% %             end
            
% %             filename = strcat('test0512GapFold',num2str(k));
% %             load(filename)
% %             load('CONST.mat')
% %             
% %             wgap3  =  gap + weight * gCoef3;
% %             ID = sign(wgap3(:,1) - wgap3(:,2)- CONST(k))/2 +3/2;
% %             ID = ID';
% %             ACC(k) = sum(ID == Ls_1)/length(Ls_1);           
            
% %         end
        
        
        %% Sparse Classification II
% %             filename = strcat('Dict050901_',num2str(k));
% %             load(filename);
             
% %             lambda = [LAMBDA(j)];
% %             [LABEL1,C1,DIF1] = SLEP_LeastR_SparseClassify_Evan1(Dict1,Drls1,Xt_1,Lt_1,lambda);
% %             filename = strcat('05121GapFold',num2str(k));
% %             save(filename,'DIF1')


% %             filename = strcat('05121GapFold',num2str(k));
% %             load(filename)

% %             
% %             gap = DIF1';
% %             
% %             Const = 0;
% %             step = 0.05;
% %             is_loop = 1;
% %             StopCri = 0.000001;
% %             nIter = 100;
% %             iter = 0;
% %             direction = +1;
% %             Lt_1_1 = Lt_1((Lt_1 == 1));
% %             Lt_1_2 = Lt_1((Lt_1 == 2));
% % 
% %             cri = gap(:,1) - gap(:,2) - Const;
% %             ID = sign(cri)/2 + 3/2;
% %             ID = ID';
% %             ID_1 = ID(Lt_1 == 1);
% %             ID_2 = ID(Lt_1 == 2);
% % 
% % 
% %     %         fprintf('%s%8f\n','reco_rate  =  ',sum(ID==Lt_1)/(length(Lt_1)));
% %             ACC_c1 = sum(ID_1 == Lt_1_1)/length(Lt_1_1);
% %             ACC_c2 = sum(ID_2 == Lt_1_2)/length(Lt_1_2);
% %             ACC_all = sum(ID==Lt_1)/(length(Lt_1));
% %             ACCmax = ACC_all;
% % 
% % 
% %             while(iter <= nIter)
% % 
% %                 Const = Const + direction * step;
% %                 cri = gap(:,1) - gap(:,2) - Const;
% %                 ID = sign(cri)/2 + 3/2;
% %                 ID = ID';
% %                 ID_1 = ID(Lt_1 == 1);
% %                 ID_2 = ID(Lt_1 == 2);            
% %                 ACC_temp = sum(ID == Lt_1)/length(Lt_1)
% %                 ACC_c1_temp = sum(ID_1 == Lt_1_1)/length(Lt_1_1);
% %                 ACC_c2_temp = sum(ID_2 == Lt_1_2)/length(Lt_1_2);
% % 
% %                 if abs(ACC_temp - ACCmax) < StopCri
% %                     is_loop = 0;
% %                 end
% % 
% %                 if ACC_temp > ACCmax
% %                     ACCmax = ACC_temp;
% %                     ACC(k) = ACCmax;
% %                     CONST(k) = Const;
% %                 end
% % 
% %                 if ACC_temp > ACC_all
% %                     ACC_all = ACC_temp;
% %                     ACC_c1 = ACC_c1_temp;
% %                     ACC_c2 = ACC_c2_temp;
% % 
% %                     continue;
% %                 elseif ACC_c1_temp <= ACC_c1 && ACC_c2_temp >= ACC_c2
% %                     direction = 1;
% %                     step = step * 0.9;
% % 
% %                     ACC_all = ACC_temp;
% %                     ACC_c1 = ACC_c1_temp;
% %                     ACC_c2 = ACC_c2_temp;
% %                 elseif ACC_c1_temp >= ACC_c1 && ACC_c2_temp <= ACC_c2
% %                     direction = -1;
% %                     step = step * 0.9;
% % 
% %                     ACC_all = ACC_temp;
% %                     ACC_c1 = ACC_c1_temp;
% %                     ACC_c2 = ACC_c2_temp;
% % 
% %                 end
% % 
% %                 iter = iter + 1;           
% %             end
% %             save('CONST2','CONST')

             
% %             lambda = [LAMBDA(j)];
% % % %             [LABEL1,C1,DIF1] = SLEP_LeastR_SparseClassify_Evan1(Dict1,Drls1,Xs_1,Ls_1,lambda);
% % % %             filename = strcat('test05131GapFold',num2str(k));
% % % %             save(filename,'DIF1')
% % % %             [LABEL2,C2,DIF2] = SLEP_LeastR_SparseClassify_Evan1(Dict2,Drls2,Xs_1,Ls_1,lambda);
% % % %             filename = strcat('test05132GapFold',num2str(k));
% % % %             save(filename,'DIF2')
% %             filename = strcat('test05131GapFold',num2str(k));
% %             load(filename)
% %             save('05134Gap.mat','DIF1all','Ls_1_all')

% %             filename = strcat('test05132GapFold',num2str(k));
% %             load(filename)
% %             load('CONST1.mat')
% %             CONST1 = CONST;
% %             load('CONST2.mat')
% %             CONST2 = CONST;
% %             

            gap1 = sign(DIF1(1,:) - DIF1(2,:) - CONST(k) )/2 + 3/2;
% %             gap2 = sign(DIF2(1,:) - DIF2(2,:) - CONST2(k) )/2 + 3/2;
% %             gap12 = sign(DIF1(1,:) - DIF1(2,:) + DIF2(1,:) - DIF2(2,:) - CONST1(k) - CONST2(k))/2 + 3/2;
            ACC(1,k) = sum(gap1 == Ls_1)/length(Ls_1);
% %             ACC(2,k) = sum(gap2 == Ls_1)/length(Ls_1);
% %             ACC(3,k) = sum(gap12 == Ls_1)/length(Ls_1);


        %% Sparse Classification III
% ------------pre-compute for Modified DALM  ----------------------------%
% %         filename = strcat('Dict050901_',num2str(k));
% %         load(filename);
% %         opts.nClass = 2;
% %         tau = 0.001;
% %         lambda = 0.1;
% % 
% %         for indClass  =  1:opts.nClass
% %             eye_M     =      eye(sum(Drls1==indClass));
% %             A         =      [Dict1(:,Drls1==indClass);sqrt(tau)*eye_M];
% %             [m,n]     =      size(A);
% %             norm_b    =      mean(sum(abs(A)));
% %             beta      =      norm_b/m;
% %             G{indClass}    = single(A * A' + sparse(eye(m)) * lambda / beta);
% %             invG{indClass} = inv(G{indClass});
% %         end
% % 
% %         ID    =  [];
% %         idf       =  [];
% %         for indTest = 1:size(Xs_1,2)
% %             fprintf(['Totalnum:' num2str(size(Xs_1,2)) 'Nowprocess:' num2str(indTest) '\n']);
% %             for indClass  =  1:opts.nClass
% %                 eye_M     =      eye(sum(Drls1==indClass));
% %                 D         =      [Dict1(:,Drls1==indClass);sqrt(tau)*eye_M];
% %                 y         =      [Xs_1(:,indTest);sqrt(tau)*CoefM1((indClass==Drls1))];
% %                 [s, nIter] = SolveDALM_M(G{indClass},invG{indClass}, D, y, 'lambda',lambda,'tolerance',1e-3);
% %                 zz        =  y - D *s;
% %                 gap(indClass)     =  zz(:)'*zz(:);
% %                 gCoef3(indClass)  =  sum(abs(s));
% %             end
% %             MixG = lambda*gCoef3+gap;
% %             index              =  find(MixG==min(MixG));
% %             par.ID(indTest)    =  index(1);
% %         end
% %         reco_ratio       =  (sum(par.ID==ttls))/length(ttls); 
% %         disp(['The recognition rate is ' num2str(reco_ratio)]); 
    end
end
