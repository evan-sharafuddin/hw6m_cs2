clear, close all

%% three channel rgb communication system
close all
image = imread("Jason-Trobaugh.jpg");

imager = image(:,:,1); imageg = image(:,:,2); imageb = image(:,:,3);
image_binr = dec2bin(imager);
image_bing = dec2bin(imageg);
image_binb = dec2bin(imageb);

% convert bits from char to int
image_binr_int = image_binr - '0';
image_bing_int = image_bing - '0';
image_binb_int = image_binb - '0';

% flatten the matrix -- transpose to maintain proper ordering
image_binr_flat = reshape(image_binr_int', [numel(image_binr_int) 1]);
image_bing_flat = reshape(image_bing_int', [numel(image_bing_int) 1]);
image_binb_flat = reshape(image_binb_int', [numel(image_binb_int) 1]);
% change all bits with value 0 to -1
image_binr_flat(image_binr_flat == 0) = -1;
image_bing_flat(image_bing_flat == 0) = -1;
image_binb_flat(image_binb_flat == 0) = -1;

% PARAMETERS TO ADJUST
N = numel(image_binr_flat); 
Tp = 0.05; % symbol width (centered around zero)
fb = 1/(2*Tp); % bit rate
sigma = 1; % noise parameter 
Ts = 1/fb; 
wcPulse = 10; % note: sinc() in matlab multiplies your input by pi
base = 20;
bandwidth = 7;
wc1 = base*2*pi; % frequency of upconverter -- currently 20 Hz
wc2 = (base+bandwidth)*2*pi; % frequency of upconverter -- currently 30 Hz
wc3 = (base+2*bandwidth)*2*pi; % frequency of upconverter -- currently 40 Hz

[~, dt, pulse] = sinc_pulse(wcPulse, Tp);


bitsr = image_binr_flat;
bitsg = image_bing_flat;
bitsb = image_binb_flat;
[ty, yr] = pam(fb, dt, Tp, Ts, N, bitsr, pulse);
[ty, yg] = pam(fb, dt, Tp, Ts, N, bitsg, pulse);
[ty, yb] = pam(fb, dt, Tp, Ts, N, bitsb, pulse);

upconverted1 = upconvert(wc1, ty, dt, yr);
upconverted2 = upconvert(wc2, ty, dt, yg);
upconverted3 = upconvert(wc3, ty, dt, yb);
upconvertedTotal = upconverted1 + upconverted2 + upconverted3;

[ynoise, noise] = addNoise(upconvertedTotal, sigma);

xhatr = downconvert(wc1, ty, dt, ynoise, pulse, N, Ts, Tp);
xhatg = downconvert(wc2, ty, dt, ynoise, pulse, N, Ts, Tp);
xhatb = downconvert(wc3, ty, dt, ynoise, pulse, N, Ts, Tp);

% change values that are -1 back to "bit values" (i.e. 0 and 1)

xnr = xhatr(xhatr ~= 0);
xng = xhatg(xhatg ~= 0);
xnb = xhatb(xhatb ~= 0);

xnr(xnr == -1) = 0;
xng(xng == -1) = 0;
xnb(xnb == -1) = 0;

bits_reshapedr = reshape(xnr, length(image_binr(1,:)),length(image_binr(:,1)));
bits_reshapedg = reshape(xng, length(image_binr(1,:)),length(image_binr(:,1)));
bits_reshapedb = reshape(xnb, length(image_binr(1,:)),length(image_binr(:,1)));

bits_reshapedr(bits_reshapedr==-1) = 0;
bits_reshapedg(bits_reshapedg==-1) = 0;
bits_reshapedb(bits_reshapedb==-1) = 0;

bits_reshapedr = char(bits_reshapedr' + '0');
bits_reshapedg = char(bits_reshapedg' + '0');
bits_reshapedb = char(bits_reshapedb' + '0');

recovered_imager = reshape(uint8(bin2dec(bits_reshapedr)), size(imager));
recovered_imageg = reshape(uint8(bin2dec(bits_reshapedg)), size(imageg));
recovered_imageb = reshape(uint8(bin2dec(bits_reshapedb)), size(imageb));

recovered_image = cat(3,recovered_imager,recovered_imageg, recovered_imageb);
figure, imshow(image)
figure, imshow(recovered_image)

%% Three channel communication system with randomized bits
close all
N = 20; % number of bits
Tp = 0.05; % symbol width (centered around zero)
fb = 1/(2*Tp); % bit rate
base = 20;
bandwidth = 5; 
wc1 = base*2*pi; % frequency of upconverter -- currently 20 Hz
wc2 = (base+bandwidth)*2*pi; % frequency of upconverter -- currently 30 Hz
wc3 = (base+2*bandwidth)*2*pi; % frequency of upconverter -- currently 40 Hz

wcPulse = 10; % zeros every 0.1 sec
sigma = 1; % noise parameter 
Ts = 1/fb; 

% create symbol
[t_sinc, dt, pulse] = sinc_pulse(wcPulse, Tp);

% [t_pulse, dt, pulse] = gauss(.6, wcPulse, Tp);
% [t_pulse, dt, pulse2] = sqrt_sinc_pulse(wcPulse, Tp);

%%% figures for latex
% figure, hold on
% subplot(2,2,1)
% plot(pulse)
% title('normal truncated sinc')
% 
% subplot(2,2,2)
% plot(pulse2)
% title('sinc square rooted multiple times')
% 
% subplot(2,2,[3 4])
% 
% plot(fftshift(abs(fft(pulse))))
% hold on
% plot(fftshift(abs(fft(pulse2))))
% title('Fourier transforms of pulse shapes')
% xlim([40 62])
% legend('fft of normal sinc','fft of square rooted sinc')
%%%

% create vector of bits (at the moment, this is random)
bits = 2*((rand(1,N)<0.5)-0.5);
bits2 = 2*((rand(1,N)<0.5)-0.5);
bits3 = 2*((rand(1,N)<0.5)-0.5);

% communication system 
% bits to signal
[ty, y] = pam(fb, dt, Tp, Ts, N, bits, pulse);
[ty2, y2] = pam(fb, dt, Tp, Ts, N, bits2, pulse);
[ty3, y3] = pam(fb, dt, Tp, Ts, N, bits3, pulse);


% upconvert to desired signal
[upconverted1,trans1,f1] = upconvert(wc1, ty, dt, y);
[upconverted2,trans2,f2] = upconvert(wc2, ty2, dt, y2);
[upconverted3,trans3,f3] = upconvert(wc3, ty3, dt, y3);
upconvertTotal = upconverted1 + upconverted2 + upconverted3;

% fft
figure;
subplot(2, 1, 1), hold on
plot(f1, abs(trans1));
plot(f2, abs(trans2));
plot(f3, abs(trans3));
title("Fourier Transform of upconverted signal");
xlabel("Frequency (Hz)");
ylabel("Magnitude");
hold off

subplot(2, 1, 2), hold on
plot(f1, angle(trans1));
plot(f2, angle(trans2));
plot(f3, angle(trans3));
title("Fourier Transform of upconverted signal");
xlabel("Frequency (Hz)");
ylabel("Angle");
% add noise to simulate transmission
% [ynoise, noise] = addNoise(upconverted1, sigma);
% [ynoise2, noise2] = addNoise(upconverted2, sigma);
% [ynoise3, noise3] = addNoise(upconverted3, sigma);
[ytot, noise] = addNoise(upconvertTotal, sigma);

figure;
plot(ytot);

% downconverts recieved signal, applies matched filter to recover original
% signal and resolve noise
xhat = downconvert(wc1, ty, dt, ytot, pulse, N, Ts,Tp);
xhat2 = downconvert(wc2, ty, dt, ytot, pulse, N, Ts,Tp);
xhat3 = downconvert(wc3, ty, dt, ytot, pulse, N, Ts,Tp);

%%% use for testing without upconversion
% [ynoise, noise] = addNoise(y, sigma);
% xhat = matchFilter(pulse, ty, ynoise, N, fb);
%%%

% recovers the bits
% recovers the bits
xn = xhat(xhat ~= 0);
xn2 = xhat2(xhat2 ~= 0);
xn3 = xhat3(xhat3 ~= 0);


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
disp("error rate 1: " + string(1-(sum(xn == bits) / length(bits))))
disp("error rate 2: " + string(1-(sum(xn2 == bits2) / length(bits2))))
disp("error rate 3: " + string(1-(sum(xn3 == bits3) / length(bits3))))


figure;
subplot(2, 1, 2)
stem(xn)
title("xn -- recieved bits")
subplot(2, 1, 1)
stem(bits)
title("bits - sent/original bits")

figure;
subplot(2, 1, 2)
stem(xn2)
title("xn -- recieved bits")
subplot(2, 1, 1)
stem(bits2)
title("bits - sent/original bits")

figure;
subplot(2, 1, 2)
stem(xn3)
title("xn -- recieved bits")
subplot(2, 1, 1)
stem(bits3)
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

function [time, dt, pulse] = sqrt_sinc_pulse(w, Tp)

dt = Tp/50; % sampling frequency -- keep this constant

% creates the time vector and the pulse
t_sinc = -Tp:dt:Tp;
sincPulse = sqrt(sqrt(sqrt(sqrt(abs(sinc(w * t_sinc))))));

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

function [time, dt, pulse] = sincSquared_pulse(w, Tp)

dt = Tp/50; % sampling frequency -- keep this constant

% creates the time vector and the pulse
t_sinc = -Tp:dt:Tp;
sincPulse = sinc(w * t_sinc).*sinc(w * t_sinc);

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


% http://complextoreal.com/wp-content/uploads/2013/01/isi.pdf
% apparently good, also he referenced this in the instructions so maybe its
% good
% https://en.wikipedia.org/wiki/Pulse_shaping
% sinc, raised cosine, and gaussian commonly used
function [time, dt, pulse] = rcos_pulse(w, rolloff, Tp)

dt = Tp/50; % sampling frequency -- keep this constant


% creates the time vector and the pulse
t_sinc = -Tp:dt:Tp;
sincPulse = sinc(w*t_sinc) .* cos(2*pi*rolloff*t_sinc)./(1-(2*rolloff*t_sinc/pi).^2);

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

function [time, dt, pulse] = gauss(w, bw, Tp)

dt = Tp/50; % sampling frequency -- keep this constant


% creates the time vector and the pulse
t_sinc = -Tp:dt:Tp;
sincPulse = gauspuls(t_sinc, w, bw);

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

xn = zeros(size(tx));

% logical indexing approach
logical = mod(tx,Ts) == 0;
logical(end) = 0;
xn(logical) = bits;



%%% ORIGINAL
% tx = 0:dt:(N)*Ts;


% insert the message onto xn, put a spike every Ts seconds
% xn = zeros(size(tx));
% for i=0:N-1
%     xn(abs(tx - i * Ts) < .0001) = bits(i+1);
%     
% end


% calculate the convolution, will place the pulse at every spike
y = conv(xn, pulse);
% remake the time vector for this new function
tout = -Tp:dt:(N)*Ts + Tp;

figure;
plot(tout, y);
title("pam")
hold on, stem(tout,[zeros(1,50) xn zeros(1,50)])
%%%

end


% will pass in signal not pulse later. pulse is the signal, not a singular
% pulse
% --- changed pulse to "signal" for more clarity
function [upconverted,transform,f] = upconvert(wc, time, dt, signal)
    % multiply by a cos to upconvert
    upconverter = cos(wc*time);
    upconverted = signal.*upconverter;

    % plot
    figure;
    plot(time, upconverted);
    xlabel('time (s)'), ylabel('y(t)'), title('Upconverted signal');
    
%     % fft
    fs = 1/dt;
    transform = fft(upconverted);
    f = 0:fs/length(transform):fs-fs/length(transform);
%     
%     
%     figure;
%     subplot(2, 1, 1);
%     plot(f, abs(transform));
%     title("Fourier Transform of upconverted signal");
%     xlabel("Frequency (Hz)");
%     ylabel("Magnitude");
% 
%     subplot(2, 1, 2);
%     plot(f, angle(transform));
%     title("Fourier Transform of upconverted signal");
%     xlabel("Frequency (Hz)");
%     ylabel("Angle");
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
function xhat = downconvert(wc, time, dt, signal, pulse, N, Ts,Tp)
    downconverted= downconvertNoLowpass(wc, time, dt, signal);

    % takes the downconverted and passes it through a filter.
    xhat = matchFilter(pulse, time, downconverted, N, Ts,dt,Tp);
    figure;
    plot(xhat);

end

% implements the filter from hw 6.
%%% GET RID OF FOR LOOP
function xhat_matched = matchFilter(pulse, tsig, noisySignal, N, Ts, dt, Tp)

%%% Original
      % fplits p, then convolutes with zn
    p_negt = flip(pulse);
    zn = conv(noisySignal, p_negt, "same");
    xhat_matched = zeros(size(noisySignal));

    % logical indexing approach -- extracts the locations of each bit
    buffer = length(0:dt:Tp);
    index = buffer:Ts/dt:length(xhat_matched)-buffer;
    xhat_matched(index) = zn(index);

    for idx = index
        if zn(idx) > 0
            xhat_matched(idx) = 1;
        else
            xhat_matched(idx) = -1;
        end
    end 

end