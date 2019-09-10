%% Kalman and MLE estimation

% Author: Giacomo Romanini - use it freely, at your own risk

% version 1.0: 13 March 2018
% version 1.2:  17 April 2019. changes:
%               - fixed Xt transpose (same result but easier to read)
% version 1.4: 04 September 2019 changes:
%               - filtering and THEN predicting step, as in Durbin Koopman 2012
%               p85)
% version 2.0: 05 September 2019 changes:
%               - univariate treatment of multivariate, as in DK2012 p 156 
%
%==========================================================================
%         Kalman Filter for MLE estimation of the state space model:
%
%                        y = A*x + H*csi +  w         w~N(0,R)
%
%                        csi = F*csi(-1) + v          v~N(0,Q)
%
%
%       Dimensions: y   : 1 x 1  
%                   A   : 1 x k
%                   x   : k x 1 -> non time-varying observables (data)
%                   H   : 1 x r -> Data for timevatying parameters
%                   csi : r x 1
%                   R   : M x M
%                   F   : r x r
%                   Q   : r x r
%                   
%   where M = number of variables in the y vector
%         r = number of unobservables/states
%         k = number of observed in the X matrix
%==========================================================================


%==========================================================================
function [MLEoutput, csi, CSI, P, p, f, F, eta] = MLEkalmanDKunivariate(y, X, H, csi_not, P_not, x)


% % ------------ MLE parameters ------------

[T , M] = size(y); % number of time periods and variables in the dependent vector
r = size(H, 2); % number of states
obs = size(X, 2); % number of observables



% M*k + r + M*M  % number of parameters to be estimated
% y = Y;
% X = X;
% H = Hkal;
% x = x0;

% Setting matrices of parameters to be estimated. 
% A contains observable coefficients
% R the Mx1 diagonal varcovar matrix of the endogenous variables in the system
% Q is the (time-fixed, for now) diagonal variance of states

A = [];
for i = 1:M*obs
    A = [A', x(i)]';
end
A = reshape(A, M, obs);

R = [];
for i = M*obs+1: M*obs+M
    R = [R', x(i)^2]';
end
R = reshape(R, M, 1);

Q = [];
for i = M*obs+M+1: M*obs+M+r
    Q = [Q', x(i)^2]';
end
Q = diag(Q);





 
% Assuming independent time-varying parameters
F = eye(r);

% ------------ Kalman Recursion ------------
% cap letters: updated | small: predicted

% ------------------- Initialize --------------

% Predicted 
csi = cell(T,1);    % rx1
p = cell(T,1);      % same dimension as Q (r x r) -> number of unobserved (matrix-> T cells)

% Updated 
CSI = cell(T,M);
P = cell(T,M);

% Intermediate steps
k = cell(T,M);      % Kalman gain. dim: rxr x rxM x MxM -> rxM (in this case rx1)
eta = zeros(T,M); 
f = zeros(T,M); % MxM, same ar R


% Intialize likelihood function
iterMLE = 0;

for t = 1:T    
    
    % Looping over multivariate variables
    for i = 1:M
        
        if t ==1 
            CSI{t, 1} = csi_not;
            P{t, 1} = P_not;
        else
            CSI{t, 1} = csi{t};
            P{t, 1} = p{t};
        end


        % Filtering equations
        eta(t,i) = y(t,i) - A*X(t,:) - H(t+(i-1)*T, :)* CSI{t, i}; % forecast error for the observation equation   
        f(t,i) = H( t+(i-1)*T, :)*P{t, i}* H( t+(i-1)*T,:)' + R; % mean squared error (+ uncertainty)
        k{t, i} = P{t, i} * H(t+(i-1)*T, :)' * pinv(f(t,i)); % Kalman Gain

        % Update     
        CSI{t, i+1} = CSI{t, i} +  k{t, i} * eta(t,i);
        P{t, i+1} = P{t, i} - k{t, i} * H( t+(i-1)*T, :) * P{t, i};

        % Likelihood
        iterMLE = iterMLE + log(f(t,i)) + eta(t,i)'*pinv(f(t,i))*eta(t,i);
   
    end
    
    % Time Predict
    csi{t+1} = F * CSI{t, M+1} ; % rx1
    p{t+1} = F * P{t, M+1} * F' + Q ; %rxr
    
       

end % end of the loop
    

lnL = -(1/2*((T*log(2*pi) + iterMLE)));
   
MLEoutput = -lnL;


% Drop the first column in CSI and P, just used for the initial values of every i
% loop
CSI = CSI(:,2:end);
P = P(:,2:end);

% Drop the first time
csi = csi(2:end,:);
p = p(2:end, :);

end % end of the function