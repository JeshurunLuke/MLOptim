%% RET evolution with randomized coupling strength, and time 0
if 1
    tic 

    Nshots = 1000; % number of averages
    dt = 0.5; % time step [us]
    t_end = 30; % total evolution time [us]
    tt = [0:dt:t_end];
    
    scan = tt;

    Nscan = length(scan);
    res = zeros(Nshots,Nscan);
    
    t_range = 0; % range of t0 [us]
    
    
    for i = 1:Nshots
        
        res_temp = zeros(1,Nscan);
        Hs = hs1();
        for j = 1: Nscan

            psi = solve_se1(Hs, scan(j), [1 0]');
                     
            res_temp(j) = abs(psi(1).^2);
        end
        
        t0 = t_range*rand(1,1);
        n_pad = round(t0/dt);
        res_temp = [ones(1,n_pad) res_temp(1:end-n_pad)];
        res(i,:) = res_temp;
    end
    res = mean(res,1);
    res = 1-res;
    
    figure(123); hold on
%     plot(ts(:)/1e-6, abs(psi.^2));
    plot(scan, res,'linewidth',2);
    box on
    set(gcf,'color','white');
    set(gca,'fontsize',14);
    xlabel('t [us]');
    ylabel('N=1 KRb population');
    toc
end    

%%


% functions

function [v] = hs() % assuming distance follows a Gaussian distribution 
 
    e = 1.602176634e-19;
    a0 = 5.29177210903e-11;
    Debye = 0.39*e*a0;
    epsilon0 = 8.8541878128e-12;
    hbar = 1.05457182e-34;
    Dryd = 8000*a0*e;
    DKRb = 0.56*Debye;
    nov = 8.84e17;
    amu = (1/nov)^(1/3);
    asigma = 0.1*amu;
%     disp(amu)
    a = normrnd(amu,asigma,1,1);
    rabi = Dryd*DKRb/(4*pi*epsilon0*a^3)/(2*pi*hbar);
    rabi = rabi*1e-6; % rescaled energy so that time is automatically in unit of us
%     disp(rabi)
    
    
    v = rabi/2*sigmax();
end

function [v] = hs1() % 1/a^3 follows a Gaussian distribution
 
    e = 1.602176634e-19;
    a0 = 5.29177210903e-11;
    Debye = 0.39*e*a0;
    epsilon0 = 8.8541878128e-12;
    hbar = 1.05457182e-34;
    kB = 1.38064852e-23;
    Dryd = 8000*a0*e;
    DKRb = 0.56*Debye;
    nov = 8.84e17;
    amu = (1/nov)^(1/3);
    asigma = 0.2*amu;
%     disp(amu)
    a = normrnd(amu,asigma,1,1);
    T =  0.4*1e-6; % K
    m0 = 1.66*1e-27; %kg
    mKRb = 127*m0;
    wx = 2*pi*100; wy = 2*pi*240; wz = 2*pi*240;
    xsigma = sqrt(kB*T/(mKRb*wx^2)); ysigma = sqrt(kB*T/(mKRb*wy^2)); zsigma = sqrt(kB*T/(mKRb*wz^2));
%     n_mol = 8000*normrnd(0,xsigma,1,1)*normrnd(0,ysigma,1,1)*normrnd(0,zsigma,1,1);
    n_mol = 8000*1/(sqrt(2*pi)^3*xsigma*ysigma*zsigma);
%     disp(n_mol);
    a3 = n_mol;
    disp(a3)
    rabi = Dryd*DKRb/(4*pi*epsilon0)/(2*pi*hbar)*a3;
 
    rabi = rabi*1e-6; % rescaled energy so that time is automatically in unit of us
%     disp(rabi)
       
    v = rabi/2*sigmax();
end

function psi = solve_se1(Hs,ts,psi0) % direct integration
    prop = expm(-1i*Hs*ts);
    psi = prop*psi0;
end

function M = uwave_pulse(rabi, phi)
M = -rabi / 2 * (cos(phi)*sigmax() - sin(phi)*sigmay());
% M = -rabi / 2 * (cos(phi)*sigmax() );
end

function M = sigmaz()
M = [1 0; ...
    0 -1];
end

function M = sigmax()
M = [0 1; ...
    1 0];
end

function M = sigmay()
M = [0 -1i; ...
    1i 0];
end