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

function[csiS, csiSmooth, SmVarCSI, SmSdCSI] = KalmanSmootherUnivariate2(csi, CSI, p, P, eta, f, F, H )

[T , M] = size(CSI);
r = size(H, 2); % number of unobserved variables



% Store 
csiS = cell(T,1); % rx1
PSmooth = cell(T,1); % rxr

% Last Smoothed = Filtered. Start from T+1 so we don't loose any
% observation
csiS{T+1} = CSI{T};
PSmooth{T+1} = P{T};


% Initial Values
rr = zeros(r,1) ;
N = zeros(r,r) ;


% loop

for t = T:-1:2 % until 2, not 1 since cannot compute it
    
    % You start from t = T, m = M and then you go down first by M-1 etc
    % when you reach t = T and m = 1, you move to the previous period and
    % get the first value, i.e. t = T-1 and m = M
       
    for m = M:-1:1  
        
        
        L = F - (F * p{t-1} * H(t+(m-1)*T,:)' * pinv(f(t,m))) * H(t+(m-1)*T,:) ; % rxr rxr rxM MxM Mxr -> r x r; (p{t-1} * Htrpinv) is the Kalman gain
        
        N = H(t+(m-1)*T,:)' * pinv(f(t,m)) * H(t+(m-1)*T, :) + L' * N * L;  % rxM MxM Mxr + rxr rxr rxr  -> rxr
        rr = H(t+(m-1)*T,:)' * pinv(f(t,m)) * eta(t,m) + L' * rr;   % rxM 1x1 1x1 +  rxr rx1 -> rxM

    end
    
    % Smoothed values
    csiS{t} = csi{t-1} + p{t-1} * rr; % rx1 + rxr rx1 -> r x 1
    PSmooth{t} = p{t-1} - p{t-1}* N * p{t-1}; % rxr rxr rxr  -> r x r
    
    % Time Update. Index is t so that you can call it with t
    rr = F' * rr ;
    N = F' * N * F ;
    
    
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