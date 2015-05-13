function D    =    FDDL_INID(data,nCol,wayInit,dictnums)
% ========================================================================
% Dictionary Initialization of FDDL, Version 1.0
% Copyright(c) 2011  Meng YANG, Lei Zhang, Xiangchu Feng and David Zhang
% All Rights Reserved.
%
% -----------------------------------------------------------------------    
%   
% Input :   (1) data :  the data matrix 
%           (2) nCol :  the number of dictioanry's columns
%           (3) wayInit:  the method to initialize dictionary
% 
% Outputs : (1) D  :    the initialized dictionary
%
%------------------------------------------------------------------------

[m,n]   =    size(data);

switch lower(wayInit)
    case {'pca'}
        [D,disc_value,Mean_Image]   =    Eigenface_f(data,nCol-1);
%         [D,disc_value,Mean_Image]   =    Eigenface_f(data,dictnums-1);
        D                           =    [D Mean_Image./norm(Mean_Image)];
    case {'random'}
        vector = randperm(n);
        D = zeros(m,dictnums);
        for i = 1:dictnums
            D(:,i) = data(:,vector(i));
        end
        D = D./repmat(sqrt(sum(D.*D,1)),m,1);
%         phi                         =    randn(m, nCol);
%         phinorm                     =    sqrt(sum(phi.*phi, 1));
%         D                           =    phi ./ repmat(phinorm, m, 1);
    case {'adjustable random'}
%         index                       =    randi(nCol,atomnums);    
        phi                         =    randn(m, dictnums);
        phinorm                     =    sqrt(sum(phi.*phi, 1));
        D                           =    phi ./ repmat(phinorm, m, 1);        
    otherwise 
        error{'Unkonw method.'}
end
return;
