%% demo - Sinc pulse shape
Ts = .1; % symbol period (rate 1/Ts)
dt = .01; % sample period
t = -5*Ts:dt:5*Ts; % time vector
x = sinc(t/Ts); % define sinc, note Matlab convention sinc(x) =
%sin(pi*x)/(pi*x)
figure
subplot(2,1,1), plot(t,x)
xlabel('time (s)'), ylabel('x(t)'), title('Truncated sinc')
fs = 1/dt; % sample frequency
Nfft = 1024; % length of fft
f = [0:fs/Nfft:fs-fs/Nfft];
subplot(2,1,2), plot(f,abs(fft(x,Nfft)))
xlabel('frequency (Hz)'), ylabel('|X(j\omega)|')
%% modulated sinc
wc = 2*pi*20; % 20Hz modulation
y = x.*cos(wc*t);
figure
subplot(2,1,1), plot(t,y)
xlabel('time (s)'), ylabel('y(t)'), title('Modulated sinc')
fs = 1/dt; % sample frequency
Nfft = 1024; % length of fft
f = [0:fs/Nfft:fs-fs/Nfft];
subplot(2,1,2), plot(f,abs(fft(y,Nfft)))
xlabel('frequency (Hz)'), ylabel('|Y(j\omega)|')
%% demodulated
xr = y.*cos(wc*t);
figure
subplot(2,1,1), plot(t,xr)
xlabel('time (s)'), ylabel('x_r(t)'), title('Demod (without LPF)')
fs = 1/dt; % sample frequency
Nfft = 1024; % length of fft
f = [0:fs/Nfft:fs-fs/Nfft];
subplot(2,1,2), plot(f,abs(fft(xr,Nfft)))
xlabel('frequency (Hz)'), ylabel('|X_r(j\omega)|')
% then LPF, e.g., lowpass()