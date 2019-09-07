% Kalman Smoother for UNIVARIATE Kalman Filter as in Durbin Koopman 2012.
% tested to work for univariate filter set as in DK2012, i.e. update step
% first and predict later


% r = dimension of the state vector 
% M = dimension of the endogenous vector


% INPUTS
% Need to run MLEkalmanDKunivariate (the forward filter) to obtain the inputs csi to f.
% Ht is the matrix of observables, F is specified as identity matrix of
% size r in the main program


% OUTPUT:
% csiSmooth : Txr smoothed series
% SmVarCSI: Txr variances of the smoothed series
% SmSdCSI = square root of SmVARCSI

function[csiSmooth, SmVarCSI, SmSdCSI] = KalmanSmootherUnivariate(CSI, p, P, eta, f, F, Ht )


T = size(CSI, 1);
r = size(Ht, 2); % number of unobserved variables



% Store
csiS = cell(T+1,1); % rx1
PSmooth = cell(T+1,1); % rxr


L = cell(T,1); % each L is rxr, as F
W = cell(T,1);
rr = cell(T,1);



% Last Smoothed = Filtered. Start from T+1 so we don't loose any
% observation

csiS{T+1} = CSI{T};
PSmooth{T+1} = P{T};

% Initial Values
rr{T} = zeros(r,1) ;
W{T} = zeros(r,r) ;


% loop
t = T-1;
while t > 0
    

    Httrpinv = Ht(t+1,:)' * pinv(f(t+1,1)); % r x M * M x M -> r x M

    L{t+1} = F - (F * p{t} * Httrpinv) * Ht(t+1,:) ; % rxr rxr rxM Mxr -> r x r; (p{t-1} * Httrpinv) is the Kalman gain
    W{t} = Httrpinv * Ht(t+1, :) + L{t+1}' * W{t+1} * L{t+1}; %rxM Mxr + rxr rxr rxr  -> rxr
    rr{t} = Httrpinv * eta(t+1,1) + L{t+1}' * rr{t+1}; 

    csiS{t+1} = CSI{t} + p{t} * rr{t}; % rx1 + rxr rx1 -> r x 1
    PSmooth{t+1} = p{t} - p{t}* W{t} * p{t}; % rxr rxr rxr  -> r x r
  

    
    t = t-1;    
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