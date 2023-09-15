%%
%AliAkbar Mahmoodzadeh 98106904 HW1 
%% ####### PART 1.A ###### %%
% Set parameters
r = 100; % Firing rate (spikes per second)
dt = 0.001; % Time step (s)
T = 1; % Simulation time (s)

% Initialize spike train vector
spikes = zeros(1, T/dt);

% Generate spikes
for t = 1:length(spikes)
    if rand() < r*dt % Poisson process with rate r and time step dt
        spikes(t) = 1;
    end
end

% Plot spike train
plot(0:dt:T-dt, spikes);
xlabel('Time (s)');
ylabel('Spikes');
ylim([-0.1 1.1]);

%% ALSO WE CAN USE THIS COMMAND WITH HINT SIMULATION THAT SUGGESTAED

% Define parameters
fr = 100;     % Firing rate (spikes per second)
tSim = 1;     % Simulation time (s)
nTrials = 100;  % Number of trials

% Generate spike train matrix and time vector using poissonSpikeGen function
[spikeMat, tVec] = poissonSpikeGen(fr, tSim, nTrials);

% Plot raster using plotRaster function
plotRaster(spikeMat, tVec);
xlabel('Time (s)');
ylabel('Trial');
title(sprintf('Poisson Spike Train Raster: FR = %d Hz, %d Trials', fr, nTrials));


%% ####### PART 1.B ###### %%
% Define parameters
fr = 100;      % Firing rate (spikes per second)
tSim = 1;      % Simulation time (s)
nTrialsVec = [100, 1000, 5000, 10000]; % Vector of number of trials to test

% Create figure for subplots
figure;

% Loop over each value of nTrials and create subplot
for i = 1:length(nTrialsVec)
    nTrials = nTrialsVec(i);

    % Generate spike train matrix using poissonSpikeGen function
    [spikeMat, ~] = poissonSpikeGen(fr, tSim, nTrials);

    % Calculate spike count probability histogram and theoretical Poisson spike count density
    count = sum(spikeMat, 2);
    x = 5:140;
    %normalixe the h by divide to nTrials
    h = hist(count, x, 'k') / nTrials;
    %crear poisson PDF with MATLAB's command
    pdf = poisspdf(x, fr);

    % Create subplot and plot spike count histogram and Poisson spike count density
    subplot(2, 2, i);
    bar(x, h);
    hold on;
    plot(x, pdf, 'r', 'LineWidth', 2);
    title(sprintf('Spike Count Probability %d Trials', nTrials));
    xlabel('Spike Count');
    ylabel('Probability');
    legend('Histogram of Data', 'Poisson');
    hold off;
    grid on; grid minor;
end

% Add overall plot title
suptitle('Spike Count Probability');

%% ####### PART 1.C ###### %%
fr = 100;
tSim = 1;
nTrials = [100:100:1000]; % set a range of values for nTrials
nTrials_count = length(nTrials);

figure;
for i = 1:nTrials_count
    % Generate spikes and calculate ISI for current nTrials value
    [spikeMat, tVec] = poissonSpikeGen(fr, tSim, nTrials(i));
    ISI = ISI_(spikeMat);

    % Plot ISI histogram
    subplot(2,5,i);
    h = hist(ISI, 100);
    x = linspace(1, 200);
    bar(x, h/fr/nTrials(i));
    hold on;
    x = 0:1:200;
    % generate exponential disribution (PDF)
    y = fr/1000 * exp(-fr * x / 1000);
    
    %==== plot ====%
    plot(x, y, 'linewidth', 2, 'color', 'r');
    title(sprintf('nTrials = %d', nTrials(i)), 'interpreter', 'latex');
    xlabel('ISI (ms)', 'interpreter', 'latex');
    ylabel('Probability', 'interpreter', 'latex');
    legend('Data Histogram', 'Theory $\lambda = 100$', ...
        'interpreter', 'latex');
end
suptitle('ISI Histogram for Different nTrials');

%% ####### PART 1.A2 ###### %%

%% ## 1.A2 SPIKE TRAIN FOR K ## %%
% Set parameters
r = 100; % Firing rate (spikes per second)
dt = 0.001; % Time step (s)
T = 1; % Simulation time (s)
kValues = 2:10;

% Initialize spike train matrix
spikesMat = zeros(length(kValues), T/dt);

% Generate and keep kth spikes for each value of k
for i = 1:length(kValues)
    k = kValues(i);
    spikes = zeros(1, T/dt);
    
    for t = 1:length(spikes)
        if rand() < r*dt % Poisson process with rate r and time step dt
            if mod(t, k) == 0
                spikes(t) = 1;
            end
        end
    end
    
    spikesMat(i, :) = spikes;
end

% Plot spike trains for each k value
colors = lines(length(kValues));
figure;
for i = 1:length(kValues)
    subplot(3, 3, i);
    plot(0:dt:T-dt, spikesMat(i, :), 'color', colors(i, :));
    xlabel('Time (s)');
    ylabel('Spikes');
    ylim([-0.1 1.1]);
    title(sprintf('Poisson Spike Train with k = %d', kValues(i)));
end
%%
% Define parameters
fr = 100;     % Firing rate (spikes per second)
tSim = 1;     % Simulation time (s)
nTrials = 100;  % Number of trials

% Generate spike train matrix and time vector using poissonSpikeGen function
[spikeMat, tVec] = poissonSpikeGen(fr, tSim, nTrials);

% Plot the spike count and ISI histograms for k = 2 to 10 with subplots
figure;
for k = 2:10
    % Delete all but every kth spike
    spikeMat_k = removeKthSpike(spikeMat, k);

    % Calculate spike count histogram
    spikeCount = sum(spikeMat_k, 2);
    [N, edges] = histcounts(spikeCount, 'BinMethod', 'integers');

    % Calculate ISI histogram
    ISI = ISI_(spikeMat_k);
    edges_ISI = linspace(0, max(ISI), 50);
    N_ISI = histcounts(ISI, edges_ISI);

    % Plot spike count histogram
    subplot(3, 3, k-1);
    stem(edges(1:end-1), N);
    xlabel('Spike Count');
    ylabel('Count');
    title(sprintf('k = %d', k));

   
end
figure;
for k = 2:10
    % Delete all but every kth spike
    spikeMat_k = removeKthSpike(spikeMat, k);

    % Calculate spike count histogram
    spikeCount = sum(spikeMat_k, 2);
    [N, edges] = histcounts(spikeCount, 'BinMethod', 'integers');

    % Calculate ISI histogram
    ISI = ISI_(spikeMat_k);
    edges_ISI = linspace(0, max(ISI), 50);
    N_ISI = histcounts(ISI, edges_ISI);

  

    % Plot ISI histogram
    subplot(3, 3, k-1);
    bar(edges_ISI(1:end-1), N_ISI);
    xlabel('ISI (ms)');
    ylabel('Count');
    title(sprintf('k = %d', k));
end



%% ####### PART 1.D ###### %%

%Calculate the CV for two data set Poisson spike and renewal Poisson (Kth)
fr = 100; % Firing rate (spikes per second)
tSim = 1; % Simulation time (s)
nTrials = 10000; % Number of trials
kValues = 1:10; % k values to test
cvPoisson = zeros(size(kValues));
cvKth = zeros(size(kValues));

for i = 1:length(kValues)
    k = kValues(i);
    % Generate Poisson spike train
    [spikeMat, tVec] = poissonSpikeGen(fr, tSim, nTrials);
    % Calculate CV for Poisson spike train
    isiPoisson = ISI_(spikeMat);
    cvPoisson(i) = std(isiPoisson)/mean(isiPoisson);
    
    % Keep every kth spike
    spikeMat_k = removeKthSpike(spikeMat, k);
    % Calculate CV for kth Poisson spike train
    isiKth = ISI_(spikeMat_k);
    cvKth(i) = std(isiKth)/mean(isiKth);
end

% Plot CV values for Poisson and kth Poisson spike trains
figure;
plot(kValues, cvPoisson, '-o', 'LineWidth', 2);
hold on;
plot(kValues, cvKth, '-o', 'LineWidth', 2);
grid on 

xlabel('K', 'interpreter', 'latex');
ylabel('Coefficient of Variation (CV)', 'interpreter', 'latex');
title('CV for Poisson and kth Poisson Spike Trains', 'interpreter', 'latex');
legend('Poisson', 'kth Poisson', 'interpreter', 'latex');

%% ####### PART 1.G ###### %%

%% ### Refactory periode for T = 0, 0.2, 13 ms and  K = 1,2,3,4,51 #### %%

fr_vec = 30: 10: 1000;
tSim = [2, 2, 10];
nTrials = 100;
k_vec = [1,2,3,4,10,20,51];
t0_vec =[0.000,0.001,0.002,0.005,0.013]

% Preallocate arrays for storing the results
CV = zeros(length(k_vec), length(t0_vec), length(fr_vec));
mean_ISI = zeros(length(k_vec), length(t0_vec), length(fr_vec));

% Loop over the values of k, t0, and fr
for k_idx = 1:length(k_vec)
    k = k_vec(k_idx);
    for t0_idx = 1:length(t0_vec)
        t0 = t0_vec(t0_idx);
        for fr_idx = 1:length(fr_vec)
            fr = fr_vec(fr_idx);

            % Generate the Poisson spike train
            [spikeMat, tVec] = poissonSpikeGen(fr, tSim(1), nTrials);

            % Generate the renewal process spike train
            renewalSpikeMat = renewalGen(k, spikeMat);

            % Calculate the ISIs
            ISI = ISI_(renewalSpikeMat);

            % Shift the ISIs by t0 to account for the dead time
            ISI = ISI + t0;

            % Calculate the coefficient of variation (CV)
            CV(k_idx, t0_idx, fr_idx) = std(ISI) / mean(ISI);
            mean_ISI(k_idx, t0_idx, fr_idx) = mean(ISI);
        end
    end
end

% Calculate the theoretical CV values for comparison
theoreticalCV = 1 ./ sqrt(k_vec);

% Plot the CV values for different values of k and t0
figure;
for k_idx = 1:length(k_vec)
    k = k_vec(k_idx);
    for t0_idx = 1:length(t0_vec)
        t0 = t0_vec(t0_idx);
        
        % Select the data for the current k and t0
        CV_data = squeeze(CV(k_idx, t0_idx, :));
        mean_ISI_data = squeeze(mean_ISI(k_idx, t0_idx, :));
        
        % Create a subplot
        subplot(length(k_vec), length(t0_vec), (k_idx-1)*length(t0_vec) + t0_idx);
        
        % Plot the theoretical CV value
        plot(1000./fr_vec, ones(size(fr_vec))*theoreticalCV(k_idx), 'p--');
        grid on 
        hold on;
        
        % Plot the empirical CV values
        plot(1000./fr_vec, CV_data, 'o');
        
        % Set the plot properties
        title(['k = ' num2str(k) ', Refactory Periode = ' num2str(t0*1000) ' ms'], 'interpreter', 'latex');
        xlabel('$\Delta$t (msec)', 'interpreter', 'latex');
        ylabel('CV', 'interpreter', 'latex');
        ylim([0 1.2]);
    end
end







%% ### Refactory periode plot like figure 6 in article #### %%
fr_vec = 30:10:1000;
tSim = [2, 2, 10];
nTrials = 100;
k_vec = [1,2,3,4,10,20,51];
t0_vec =[0.000,0.001,0.002,0.005,0.013]



% Preallocate arrays for storing the results
CV = zeros(length(k_vec), length(t0_vec), length(fr_vec));
mean_ISI = zeros(length(k_vec), length(t0_vec), length(fr_vec));

% Loop over the values of k, t0, and fr
for k_idx = 1:length(k_vec)
    k = k_vec(k_idx);
    for t0_idx = 1:length(t0_vec)
        t0 = t0_vec(t0_idx);
        for fr_idx = 1:length(fr_vec)
            fr = fr_vec(fr_idx);

            % Generate the Poisson spike train
            [spikeMat, tVec] = poissonSpikeGen(fr, tSim(1), nTrials);

            % Generate the renewal process spike train
            renewalSpikeMat = renewalGen(k, spikeMat);

            % Calculate the ISIs
            ISI = ISI_(renewalSpikeMat);

            % Shift the ISIs by t0 to account for the dead time
            ISI = ISI + t0;

            % Calculate the coefficient of variation (CV)
            CV(k_idx, t0_idx, fr_idx) = std(ISI) / mean(ISI);
            mean_ISI(k_idx, t0_idx, fr_idx) = mean(ISI);
        end
    end
end

% Calculate the theoretical CV values for each k
theoreticalCV = 1 ./ sqrt(k_vec);

% Plot the CV values for different values of k and t0
for t0_idx = 1:length(t0_vec)
    t0 = t0_vec(t0_idx);
    figure;
    for k_idx = 1:length(k_vec)
        k = k_vec(k_idx);
        
        % Calculate the theoretical CV value for the current k
        theoreticalCV_k = ones(size(fr_vec)) * theoreticalCV(k_idx);
        
        % Select the data for the current k and t0
        CV_data = squeeze(CV(k_idx, t0_idx, :));
        mean_ISI_data = squeeze(mean_ISI(k_idx, t0_idx, :));
        
        % Plot the theoretical CV value
        plot(1000./fr_vec, theoreticalCV_k, 'k--');
        grid on 
        grid minor 
        hold on;
        
        % Plot the empirical CV values
        plot(1000./fr_vec, CV_data, 'o');
        
        % Set the plot properties
        title(['k = '  ', Refactory Periode = ' num2str(t0*1000) ' ms'], 'interpreter', 'latex');
        xlabel('$\Delta$t (msec)', 'interpreter', 'latex');
        ylabel('CV', 'interpreter', 'latex');
        ylim([0 1.2]);
        if k_idx == 1
            legend('Theoretical', 'Empirical', 'Location', 'northwest', 'interpreter', 'latex');
        end
    end
end

%% ###### QUESTION @2 ###### %%


%% == Leaky Integrate and Fire Neuron == %%


%% ###### PART A ###### %%
clear; clc; close all;
vr = 0;
Tm = 13; %msec as used in the paper
I = 20; %mv
v(1) = vr;
dt = 0.001; %ms
vth = 15; %mv
time = 0: dt: 100; %msec
s = 0; %spike count
for i = 1: 1: length(time)-1
    dv(i) = (1/Tm)*(-v(i) + I);
    v(i+1) = v(i) + dt * dv(i);
    if (v(i+1) > vth)
        v(i+1) = 0;
        
    end
end

for i = 1: 1: length(v)-1
    
    if (v(i) == 14.960561633490169)
        
     v(i) = 17
    end
end
vnew = v(1:length(v)-1)

figure;
plot(time, v);
title('Membrane Potential','interpreter', 'latex');
xlabel('time(ms)', 'interpreter', 'latex');
ylabel('Membrane Voltage(mv)','interpreter', 'latex');
ylim([0 17]);



%% ==== PART 2.C GENERATE Current === %%:
clear; clc; close all;
fr = 200; %as what Softky and Koch did in Fig. 8
tSim = 1; %s
nTrials = 10;
I = inputGen(fr, tSim, nTrials);

% generating v:
vr = 0;
Tm = 13; %msec
vth = 15; %mv
vth = 0.025


figure;

plot(I)
grid on 
grid minor
title('I(t)', 'interpreter', 'latex');
xlabel('time(msec)', 'interpreter', 'latex');
xlim([0 150]);

%%== PART 2.C GENERATE Voltage === %%:

dt = 0.0001; % Simulation time step
Duration = 20; % Simulation length
T = (Duration/dt);
t = (1:T) * dt; % Simulation time points in ms



tau_m = 13; 
tau_peak = 1.5;
Kernel = t .* exp(-t/tau_peak); 
v = 0 * ones(1,T); 
R = 1; 
I = ones(1,T);
dv = 0;
boolean = 0;
vth = 0.025;


% Euler method for v(t)
for i = 1:(T-1)
    
    if (v(i) < vth)
        dv = (-v(i) + R*I(i)) / tau_m;
        v(i+1) = v(i) + dv*dt;
        boolean = 0;
    
    elseif (v(i) >= vth) 
        if(boolean == 1)
            v(i) = 0;
        else
            dv = (-v(i) + 0.04)^(0.5);
            v(i+1) = v(i) + dv*dt;
            if(v(i) >= 0.035)
                boolean = 1;
            end
        end
    end
end

figure;

plot(t,v);
xlabel('Time(ms)','interpreter','latex');
ylabel('Voltage(V)','interpreter','latex');
grid on; grid minor;
title('Voltage vs Time','interpreter','latex');


%% contour plot
close all; clc; clear;
fr = 200; 
tSim = 1; %s
nTrials = 10;
vr = 0;
Tm_vec = 0.1: 0.1: 10; %msec
nth_vec = 1: 5: 100;
for j = 1: 1: length(Tm_vec)
    Tm = Tm_vec(j);
    for i = 1: 1: length(nth_vec)
        nth = nth_vec(i);
        I = inputGen(fr, tSim, nTrials);
        vth(j, i) = nth * max(I)/2000;
        [v, vSpike, freq] = actionPotential(I, Tm, vr, vth);
        ISI = find_ISI(vSpike);
        cv(j, i) = std(ISI)/mean(ISI);
    end
end
figure;
contour(Tm_vec, nth_vec, cv', 8, 'linewidth', 2);
title('Contour Plot of CV', 'interpreter', 'latex');
xlabel('$T_m$', 'interpreter', 'latex');
ylabel('$N_{th}$', 'interpreter', 'latex');
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')


%% == part E:
clear; clc; close all;
M = 100;
nTrials = M;
tSim = 1;
fr = 200;
[spikeMat, tVec] = poissonSpikeGen(fr, tSim, nTrials);
R_vec = [0.1: 0.02: 1];
D_vec = [1: 1: 50]; %msec
for d = 1: 1: length(D_vec)
    D = D_vec(d);
    for r = 1: 1: length(R_vec)
        R = R_vec(r);
        N = R * M;
        spikeOut = zeros(1, 1000);
        for i = 1: 1: (1000/D)
            if (D*i < 1000)
                mat = spikeMat(:, [D*i - (D-1): D*i]);
                c = 0;
                for j = 1: 1: M
                    if(length(find(mat(j, :))))
                        c = c + 1;
                    end
                end
                if(c > N)
                    spikeOut(i) = 1;
                end
            else
                break;
            end
            ISI = find_ISI(spikeOut);
            CV_part_e(d, r) = std(ISI)/mean(ISI);
        end
    end
end

figure;
imagesc(CV_part_e);
xlabel('R=$\frac{M}{N}$ ratio', 'interpreter', 'latex');
ylabel('D(msec)', 'interpreter', 'latex');
%% ###### FUNCTION ###### %%
function [] = plotRaster(spikeMat, tVec)
    hold all;
    for trialCount = 1:size(spikeMat,1)
         spikePos = tVec(spikeMat(trialCount, :));
         for spikeCount = 1:length(spikePos)
             plot([spikePos(spikeCount) spikePos(spikeCount)], ...
             [trialCount-0.4 trialCount+0.4], 'k');
         end
    end
    ylim([0 size(spikeMat, 1)+1]);
end

function [spikeMat, tVec] = poissonSpikeGen(fr, tSim, nTrials)
    dt = 1/1000; % s = 1 ms
    nBins = floor(tSim/dt);
    spikeMat = rand(nTrials, nBins) < fr*dt;
    tVec = 0:dt:tSim-dt;
end

function [ISI] = ISI_(spikeMat)
    nTrials = size(spikeMat, 1);
    ISI = zeros(nTrials, 150);
    for i = 1: 1: nTrials
        ISI(i, [1: 1: length(diff(find(spikeMat(i, :))))]) = ...
            diff(find(spikeMat(i, :)));
    end
    ISI = reshape(ISI, [], 1);
    ISI = ISI(find(ISI));
end

function spikeMat_k = removeKthSpike(spikeMat, k)
% Removes every kth spike in each trial of a spike matrix
% spikeMat: spike matrix with dimensions (nTrials x nBins)
% k: index of spike to remove (default value is 2)

if nargin < 2
    k = 2;
end

[nTrials, nBins] = size(spikeMat);
spikeMat_k = spikeMat;
for i = 1:nTrials
    spikes = find(spikeMat(i,:));
    if length(spikes) >= k
        removeSpikes = spikes(mod(1:length(spikes), k) == 0);
        spikeMat_k(i, removeSpikes) = 0;
    end
end


end

function [renewalSpikeMat] = renewalGen(k, spikeMat)
    nTrials = size(spikeMat, 1);
    renewalSpikeMat = zeros(nTrials, size(spikeMat, 2));
    for i = 1: 1: nTrials
        v = find(spikeMat(i, :));
        v = v([1: k: end]);
        renewalSpikeMat(i, [v]) = 1;
    end
end

function [I] = inputGen(fr, tSim, nTrials)
    
    tPeak = 1.5; %msec
    [ISpikes, tVec] = poissonSpikeGen(fr, tSim, nTrials);
    ISpikes= reshape(ISpikes, [], 1);
    Is = tVec * 1000 .* exp(-tVec ./ tPeak * 1000);   
    I =  conv(ISpikes, Is, 'same');
end