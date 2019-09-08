% Kalman Smoother for UNIVARIATE Kalman Filter as in Durbin Koopman 2012.
% tested to work for univariate filter set as in DK2012, i.e. update step
% first and predict later


% r = dimension of the state vector 
% M = dimension of the endogenous vector


% INPUTS
% Need to run MLEkalmanDKunivariate (the forward filter) to obtain the inputs csi to f.
% Ht is the matrix of observables, F is specified as identity matrix of
% dimension r = size of the state, in the main program


% OUTPUT:
% csiSmooth : Txr smoothed series
% SmVarCSI: Txr variances of the smoothed series
% SmSdCSI = square root of SmVARCSI

function[csiSmooth, SmVarCSI, SmSdCSI] = KalmanSmootherUnivariate2(csi, CSI, p, P, eta, f, F, H )

[T , M] = size(CSI);
r = size(H, 2); % number of unobserved variables



% Store: T+1 since it's a backward smoother and therefore we would loose
% the first observation since matlab doesn't like index=0. To avoid the
% issue I create a T+1 array and then remove the first (empty) 
csiS = cell(T+1,M); % rx1
PSmooth = cell(T+1,M); % rxr

% Last Smoothed = Filtered. Start from T+1 so we don't loose any
% observation
csiS{T+1} = CSI{T};
PSmooth{T+1} = P{T};


% Intermediate values, to store temp

W = cell(T,M);
rr = cell(T,M);

% Initial Values
rr{T,M} = zeros(r,1) ;
W{T,M} = zeros(r,r) ;


% loop

for t = T:-1:2 % until 2, not 1 since cannot compute it
    
    % You start from t = T, m = M and then you go down first by M-1 etc
    % when you reach t = T and m = 1, you move to the previous period and
    % get the first value, i.e. t = T-1 and m = M
       
    for m = M:-1:1
        
        % Intermediate elements
        Htrpinv = H(t,:)' * pinv(f(t,m)); % r x M * M x M -> r x M
        L = F - (F * p{t-1, m} * Htrpinv) * H(t,:) ; % rxr rxr rxM Mxr -> r x r; (p{t-1} * Htrpinv) is the Kalman gain
        
        % Dynamic elements
        W{t, m-1} = Htrpinv * H(t, :) + L' * W{t, m} * L; %rxM Mxr + rxr rxr rxr  -> rxr
        rr{t, m-1} = Htrpinv * eta(t,m) + L' * rr{t, m}; 

        % Smoothes values
        csiS{t, m} = csi{t, m-1} + p{t, m-1} * rr{t, m-1}; % rx1 + rxr rx1 -> r x 1
        PSmooth{t, m} = p{t, m-1} - p{t, m-1}* W{t, m-1} * p{t, m-1}; % rxr rxr rxr  -> r x r

    end
    
    % Time Update
    
    
    
end

% Drop the first empty observation
csiS = csiS(2:end , :) ;
PSmooth = PSmooth(2:end, :) ;


    
% Array form
csiSmooth = cell2mat(cellfun(@transpose, csiS, 'UniformOutput', false));

% The following line executes these operation (in order): 1) diagonal of
% the variance, 2) transpose, which serves for 3) array form

SmVarCSI = cell2mat(cellfun(@transpose, cellfun(@diag, PSmooth, 'UniformOutput', false), 'UniformOutput', false));
SmSdCSI = sqrt(SmVarCSI);

end