%% part 2
clear, close all

% used to test part 2
% [t_sinc, dt, sincPulse] = sinc_pulse(2*pi, .5);

% TODO: make the other pulse for part 2
%% part 3

% used to test
% wc1 = 20*2*pi;
% upconverted1 = upconvert(wc1, t_sinc, dt, sincPulse);
% downconverted1 = downconvertNoLowpass(wc1, t_sinc, dt, upconverted1);

%% converting image to bits
close all
image = imread("amongus_small.jpg");
image_gray = rgb2gray(image);
image_gray_small = image_gray;
image_bin = dec2bin(image_gray_small);

% adjusted parameters
N = numel(image_bin); 
Tp = 0.05; % symbol width (centered around zero)
fb = 1/(2*Tp); % bit rate
wc1 = 20*2*pi; % frequency of upconverter -- currently 20 Hz
sigma = 0; % noise parameter 
Ts = 0.1; 

[~, dt, pulse] = ppulse(Tp);
bits = image_bin;
[ty, y] = pam(fb, dt, Tp, Ts, N, bits, pulse);
% upconverted1 = upconvert(wc1, ty, dt, y);
% [ynoise, noise] = addNoise(upconverted1, sigma);
% xhat = downconvert(wc1, ty, dt, ynoise, pulse, N, Ts);

[ynoise, noise] = addNoise(y, sigma);
xhat = matchFilter(pulse, ty, ynoise, N, fb);

% recovers the bits
xn = xhat(xhat ~= 0);
xn = xn(1:end-1);

bits_reshaped = reshape(xn, length(image_bin(1,:)),length(image_bin(:,1)));
size(bits_reshaped)
bits_reshaped(bits_reshaped==-1) = 0;
bits_reshaped = char(bits_reshaped' + '0');
recovered_image = reshape(uint8(bin2dec(bits_reshaped)), size(image_gray));
imshow(recovered_image)

%% part 4
N = 20; % number of bits
Tp = 0.05; % symbol width (centered around zero)
fb = 1/(2*Tp); % bit rate
wc1 = 20*2*pi; % frequency of upconverter -- currently 20 Hz
sigma = 1; % noise parameter 
Ts = 0.1; 

% create symbol
% [t_sinc, dt, sincPulse] = sinc_pulse(2*pi, Tp);
[t_pulse, dt, pulse] = sinc_pulse(wc1, Tp);

% create vector of bits (at the moment, this is random)
bits = 2*((rand(1,N)<0.5)-0.5);

% communication system 
% bits to signal
[ty, y] = pam(fb, dt, Tp, Ts, N, bits, pulse);

% upconvert to desired signal
upconverted1 = upconvert(wc1, ty, dt, y);

% add noise to simulate transmission
[ynoise, noise] = addNoise(upconverted1, sigma);

% downconverts recieved signal, applies matched filter to recover original
% signal and resolve noise
xhat = downconvert(wc1, ty, dt, ynoise, pulse, N, Ts);

%%% use for testing without upconversion
% [ynoise, noise] = addNoise(y, sigma);
% xhat = matchFilter(pulse, ty, ynoise, N, fb);
%%%

% recovers the bits
xn = xhat(xhat ~= 0);



% noise level
disp("noise coeff " + sigma)
disp("bit rate " + 1/Ts)

% snr
Py = sum(y.^2 * dt);
Pn = sum(noise.^2 * dt);

if (Pn == 0)
    snr = 0;
else
    snr = Py/Pn;
end


disp("snr " + snr );

% error
disp("error rate " + string(1-(sum(xn == bits) / length(bits))))

figure;
subplot(2, 1, 2)
stem(xn)
title("xn -- recieved bits")
subplot(2, 1, 1)
stem(bits)
title("bits - sent/original bits")
% pause;

%% Functions
% function used to generate a triangle pulse. Tp is the width (in time) of
% the pulse
function [time, dt, pulse] = ppulse(Tp)
dt = Tp/50; % sampling frequency -- keep this constant

time = -Tp:dt:Tp;
pulse = 1-abs(time./Tp);
end

% generates a sinc pulse instead of a triangle. again, Tp is the width, and
% w is the angular frequency of the sinc wave.
function [time, dt, pulse] = sinc_pulse(w, Tp)

dt = Tp/50; % sampling frequency -- keep this constant

% creates the time vector and the pulse
t_sinc = -Tp:dt:Tp;
sincPulse = sinc(w * t_sinc);

% plot
figure;
plot(t_sinc, sincPulse);
title("Sinc");
xlabel("Time (s)");
ylabel("Amplitude");
fs = 1/dt;

transform = fft(sincPulse);
f = 0:fs/length(transform):fs-fs/length(transform);

figure;
subplot(2, 1, 1);
plot(f, abs(transform));
title("Fourier Transform");
xlabel("Frequency (Hz)");
ylabel("Magnitude");

subplot(2, 1, 2);
plot(f, angle(transform));
title("Fourier Transform");
xlabel("Frequency (Hz)");
ylabel("Angle");

time = t_sinc;
pulse = sincPulse;

end

% function to add a spike every Ts seconds.
%%% GET RID OF FOR LOOP
function [tout, y] = pam(fb, dt, Tp, Ts, N, bits, pulse)
tx = 0:dt:(N)*Ts;

% insert the message onto xn, put a spike every Ts seconds
xn = zeros(size(tx));
log = mod(tx,Ts) < 0.0001;
indiv_bits = bits - '0';
% need to transpose or else reshape orders it wrong
bits_flat = reshape(indiv_bits', [numel(indiv_bits) 1]);
% length of log is off for some reason
numel(bits_flat)
sum(log)
xn(log(2:end)) = bits_flat;

% calculate the convolution, will place the pulse at every spike
y = conv(xn, pulse);
% remake the time vector for this new function
tout = -Tp:dt:(N)*Ts + Tp;

figure;
plot(tout, y);
title("pam")
hold on, stem(tout,[zeros(1,50) xn zeros(1,50)])
end

% will pass in signal not pulse later. pulse is the signal, not a singular
% pulse
% --- changed pulse to "signal" for more clarity
function upconverted = upconvert(wc, time, dt, signal)
    % multiply by a cos to upconvert
    upconverter = cos(wc*time);
    upconverted = signal.*upconverter;

    % plot
    figure;
    plot(time, upconverted);
    xlabel('time (s)'), ylabel('y(t)'), title('Upconverted signal');
    
    % fft
    fs = 1/dt;
    transform = fft(upconverted);
    f = 0:fs/length(transform):fs-fs/length(transform);
    
    
    figure;
    subplot(2, 1, 1);
    plot(f, abs(transform));
    title("Fourier Transform of upconverted signal");
    xlabel("Frequency (Hz)");
    ylabel("Magnitude");

    subplot(2, 1, 2);
    plot(f, angle(transform));
    title("Fourier Transform of upconverted signal");
    xlabel("Frequency (Hz)");
    ylabel("Angle");
end

% function to add noise with a signal y and a sigma for noise.
function [ynoise, nt] = addNoise(y, sigma)
nt = sigma*randn(1,length(y));
ynoise = y + nt;
end

% function to downconvert (without the filter component)
% changed uppulse to "upconverted signal" for more clarity
function downconverted = downconvertNoLowpass(wc, time, dt, upcon_signal)
    % multiply by the cos to downconvert.
    downconverter = cos(wc*time);
    downconverted = upcon_signal.*downconverter;

    figure;
    plot(time, downconverted);
    xlabel('time (s)'), ylabel('y(t)'), title('downconverted signal');
    
    % fft
    fs = 1/dt;
    transform = fft(downconverted);
    f = 0:fs/length(transform):fs-fs/length(transform);
    
    
    figure;
    subplot(2, 1, 1);
    plot(f, abs(transform));
    title("Fourier Transform of downconverted signal");
    xlabel("Frequency (Hz)");
    ylabel("Magnitude");

    subplot(2, 1, 2);
    plot(f, angle(transform));
    title("Fourier Transform of downconverted signal");
    xlabel("Frequency (Hz)");
    ylabel("Angle");
end

% calls the above downconverter, then lowpasses it
function xhat = downconvert(wc, time, dt, signal, pulse, N, Ts)
    downconverted= downconvertNoLowpass(wc, time, dt, signal);

    % takes the downconverted and passes it through a filter.
    xhat = matchFilter(pulse, time, downconverted, N, Ts);
    figure;
    plot(xhat);

end

% implements the filter from hw 6.
%%% GET RID OF FOR LOOP
function xhat_matched = matchFilter(pulse, tsig, noisySignal, N, Ts)
    % fplits p, then convolutes with zn
    p_negt = flip(pulse);
    zn = conv(noisySignal, p_negt, "same");
    xhat_matched = zeros(size(noisySignal));

    log = mod(tsig,Ts) < 0.0001;
    log_pos = zn(log) > 0;
    log_neg = ~log_pos;
    xhat_matched(log_pos) = 1;
    xhat_matched(log_neg) = -1;
    
%     for i=0:N-1
%         index = find(abs(tsig - i* Ts) < .0001);
%           
%         if zn(index) > 0
%             xhat_matched(index) = 1;
%         else
%             xhat_matched(index) = -1;
%         end
%      end

        

    



end