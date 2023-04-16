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

%% part 4
N = 20; % number of bits
Tp = .5;
fb = 1/(2*Tp); % bit rate
wc1 = 20*2*pi;
sigma = 0;

[t_sinc, dt, sincPulse] = sinc_pulse(2*pi, Tp);
%[t_sinc, dt, sincPulse] = ppulse(Tp);
bits = 2*((rand(1,N)<0.5)-0.5);

[ty, y] = pam(fb, dt, Tp, N, bits, sincPulse);

upconverted1 = upconvert(wc1, ty, dt, y);
[ynoise, noise] = addNoise(y, sigma);

xhat = downconvert(wc1, ty, dt, ynoise, sincPulse, N, fb);
xn = xhat(xhat ~= 0);

% noise level
disp("noise coeff " + sigma)

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
disp("rate " + sum(xn == bits) / length(bits) )

figure;
subplot(2, 1, 1)
stem(xn)
title("xn")
subplot(2, 1, 2)
stem(bits)
title("bits")
pause;

%% Functions
function [time, dt, pulse] = ppulse(Tp)
dt = Tp/50; % sampling frequency -- keep this constant

time = -Tp:dt:Tp;
pulse = 1-abs(time./Tp);
end
function [time, dt, pulse] = sinc_pulse (w, Tp)
    
    % param
    dt = Tp/50; % sampling frequency -- keep this constant
    

    t_sinc = -Tp:dt:Tp;
    sincPulse = sinc(w * t_sinc);
    % creates triangular pulse that is 2*Tp wide in time
    figure;
    plot(t_sinc, sincPulse);
    title("Sinc");
    xlabel("Time (s)");
    ylabel("Amplitude");
    % plot(t_pulse,p)
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

function [tout, y] = pam(fb, dt, Tp, N, bits, pulse)
Ts = 1/fb;
tx = 0:dt:(N)*Ts;

% insert the message onto xn, put a spike every Ts seconds
xn = zeros(size(tx));
for i=0:N-1
    xn(abs(tx - i * Ts) < .0001) = bits(i+1);
end
% calculate the convolution
y = conv(xn, pulse);
tout = -Tp:dt:(N)*Ts + Tp;
figure;
plot(tout, y);
title("pam")
end

% will pass in signal not pulse later
function upconverted = upconvert(wc, time, dt, pulse)
    upconverter = cos(wc*time);
    upconverted = pulse.*upconverter;
    figure;
    plot(time, upconverted);
    xlabel('time (s)'), ylabel('y(t)'), title('Modulated sinc');
    
    % fft
    fs = 1/dt;
    transform = fft(upconverted);
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
end
function [ynoise, nt] = addNoise(y, sigma)
nt = sigma*randn(1,length(y));
ynoise = y + nt;
end
function downconverted = downconvertNoLowpass(wc, time, dt, uppulse)
    downconverter = cos(wc*time);
    downconverted = uppulse.*downconverter;

    % todo: add filter
    % xhat = matchFilter()


    figure;
    plot(time, downconverted);
    xlabel('time (s)'), ylabel('y(t)'), title('Modulated sinc');
    
    % fft
    fs = 1/dt;
    transform = fft(downconverted);
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
end
function xhat = downconvert(wc, time, dt, signal, pulse, N, fb)
    downconverted= downconvertNoLowpass(wc, time, dt, signal);

    % todo: add filter
    xhat = matchFilter(pulse, time, downconverted, N, fb);
    figure;
    plot(xhat);

end

function xhat_matched = matchFilter(pulse, tsig, noisySignal, N, fb)
    p_negt = flip(pulse);
    zn = conv(noisySignal, p_negt, "same");
    xhat_matched = zeros(size(noisySignal));
    Ts = 1/fb;
    for i=0:N-1
        index = find(abs(tsig - i* Ts) < .001);
        
    
        if zn(index) > 0
            xhat_matched(index) = 1;
        else
            xhat_matched(index) = -1;
        end

    end

end