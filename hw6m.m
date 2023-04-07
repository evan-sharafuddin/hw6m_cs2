clear, close all

% param
N = 20; % number of bits
Tp = 0.1; % half the pulse width
dt = Tp/50; % sampling frequency -- keep this constant
fb = 1/(Tp); % bit rate
dt_b = 1/fb; % bit period (how long between bits)
t = 0:dt:(dt_b*N); % time range -- not sure if this is right
sigma = 1; % noise parameter for part e

% upper limit is the number of bit periods times the number of bits

%% PART A -- need graphs, etc
t_pulse = -Tp:dt:Tp;
p = 1-abs(t_pulse./Tp);
% creates triangular pulse that is 2*Tp wide in time

% plot(t_pulse,p)

transform = fft((p(1:end-1)));
Fs = 1/dt;
f = Fs * (0:length(transform)-1)/length(transform);

% figure, stem(abs(transform))

% testing shifted vs unshifted
% x = [0:0.2:1 0.8:-0.2:0.2];
% X = fft(x); X_shift=fft(fftshift(x));
% figure,subplot(2,1,1)
% stem(angle(X))
% subplot(2,1,2), stem(angle(X_shift))
% 
% figure,subplot(2,1,1)
% stem(abs(X))
% subplot(2,1,2)
% stem(abs(X_shift))

%% PART C
bits = 2*((rand(1,N)<0.5)-0.5);

%% PART D

% Thomas
% How I think x should be generated: u take the bits defined in c, and put
% them into an array along with time. every Ts time, there will be one of
% the spikes corresponding to one in bits
% result is kinda same as urs evan, but maybe its more general? idk

% makes a time vector of appropriate length. I chose Ts arbitrarily
Ts = 1/fb;
tx = 0:dt:(N)*Ts;

% insert the message onto xn, put a spike every Ts seconds
xn = zeros(size(tx));
for i=0:N-1
    xn(abs(tx - i * Ts) < .0001) = bits(i+1);
end
% calculate the convolution
y_conv = conv(xn, p);

%% part E
% add noise as specified
nt = sigma*randn(1,length(y_conv));
rt = nt + y_conv;

%% part F

% make a new time axis, again, time is increased by Tp on both sides
tx_out = -Tp:dt:(N)*Ts + Tp;
xhat = zeros(size(rt));
xhat_matched = zeros(size(rt));

p_negt = flip(p);
zn = conv(rt, p_negt, "same");

% make the zero array then add the +1 or -1 as described in the doc
for i=0:N-1
    index = find(abs(tx_out - i* Ts) < .001);
    if rt(index) > 0
        xhat(index) = 1;
    else
        xhat(index) = -1;
    end

    if zn(index) > 0
        xhat_matched(index) = 1;
    else
        xhat_matched(index) = -1;
    end

end


%% Part g
% i
figure;
plot(t_pulse, p);
xlabel('time'),ylabel('p')
title('plot of triangular pulse p(t)')

% TODO plot fft and hand calculate fft
figure
subplot(2, 1, 1)
stem(f, abs(transform))
title('|P(j\omega)|')
xlabel('\omega')
subplot(2, 1, 2)
% correcting phase noise -- the imagionary values were extremely tiny but
% were still causing there to be nonzero phase when the phase should've
% actually been zero

true_phase = zeros(length(transform),1);

for i = 1:length(transform)
    if imag(transform(i)) < 1e-6
        true_phase(i) = angle(real(transform(i)));
    else
        true_phase(i) = angle(transform(i));
    end
end

stem(f,true_phase)
xlabel('\omega')
title('\angle P(j\omega)')
%%
% ii
figure;
plot(tx_out, y_conv)
hold on, stem(tx_out,[zeros(1, floor(length(p)/2)) xn zeros(1, floor(length(p)/2))])
title('plot of transmitted signal vs x_n')
legend('y(t)','x_n(t)')
hold off
% iii
figure;
plot(tx_out, rt);
% plot it against the original impulse message. append zeros. the
% convolution will increase size of axis by half of the size of triangle
% (Tp) on both sides, so add that on
hold on,stem(tx_out,[zeros(1, floor(length(p)/2)) xn zeros(1, floor(length(p)/2))])
title('recieved signal vs. input signal')
xlabel('time')
legend('recieved signal r(t)','input signal x_n')
hold off

% iv
figure, hold on
stem(tx, xhat(Tp/dt+1:end-Tp/dt)),stem(tx,xn)
legend('Signed-based reciever','input signal')
title('sent message and decoded message, signed-based reciever')
xlabel('time')
hold off

figure,hold on
stem(tx, xhat_matched(Tp/dt+1:end-Tp/dt)),stem(tx,xn)
legend('Matched filter reciever','input signal')
title('sent message and decoded message, matched filter reciever')
xlabel('time')
hold off

% v
% bit rate 
disp("Bit rate: " + fb)
disp("noise standard deviation (essentially 1): " + std(nt))

Py = sum(y_conv.^2 * dt);
Pn = sum(nt.^2 * dt);
disp("SNR: " + Py/Pn);

% error rate calculations
xn_sign = xhat(xhat ~= 0);
xn_matched = xhat_matched(xhat_matched ~= 0);
error_rate_sign = sum(xn_sign ~= bits)/N;
error_rate_matched = sum(xn_matched ~= bits)/N;

disp("sign-based reciever error rate: " + error_rate_sign);
disp("matched filter error rate: " + error_rate_matched);

%% part 2
% param
N = 40; % number of bits
Tp = 0.1; % half the pulse width
dt = Tp/50; % sampling frequency -- keep this constant
dt_b = 1/fb; % bit period (how long between bits)
t = 0:dt:(dt_b*N); % time range -- not sure if this is right

t_pulse = -Tp:dt:Tp;
p = 1-abs(t_pulse./Tp);
% cycle through the different fb
for inter = 1:2
    fb = 1/inter/Tp;
    sigmarray = 0:.1:1;
    snrs = zeros(size(sigmarray));
    errorSign = zeros(size(sigmarray));
    errorMatch = zeros(size(sigmarray));
    % cycle through sigmas
    for in = 1:length(sigmarray)
        % do stuff that i copied from above
        thisig = sigmarray(in);
    
        bits = 2*((rand(1,N)<0.5)-0.5);
        Ts = 1/fb;
        tx = 0:dt:(N)*Ts;
        
        % insert the message onto xn, put a spike every Ts seconds
        xn = zeros(size(tx));
        for i=0:N-1
            xn(abs(tx - i * Ts) < .0001) = bits(i+1);
        end
        % calculate the convolution
        y_conv = conv(xn, p);
        
        nt = thisig*randn(1,length(y_conv));
        rt = nt + y_conv;
        
        tx_out = -Tp:dt:(N)*Ts + Tp;
        xhat = zeros(size(rt));
        xhat_matched = zeros(size(rt));
        
        p_negt = flip(p);
        zn = conv(rt, p_negt, "same");
        
        % make the zero array then add the +1 or -1 as described in the doc
        for i=0:N-1
            index = find(abs(tx_out - i* Ts) < .001);
            if rt(index) > 0
                xhat(index) = 1;
            else
                xhat(index) = -1;
            end
        
            if zn(index) > 0
                xhat_matched(index) = 1;
            else
                xhat_matched(index) = -1;
            end
        
        end
        Py = sum(y_conv.^2 * dt);
        Pn = sum(nt.^2 * dt);
        % disp("SNR: " + Py/Pn);
        
        snrs(in) = Py/Pn;
        if Pn == 0
            snrs(in) = 0;
        end
        xn_sign = xhat(xhat ~= 0);
        xn_matched = xhat_matched(xhat_matched ~= 0);
        error_rate_sign = sum(xn_sign ~= bits)/N;
        error_rate_matched = sum(xn_matched ~= bits)/N;
        errorSign(in) = error_rate_sign;
        errorMatch(in) = error_rate_matched;
    end
    figure;
    hold on
    plot(errorSign, snrs);
    plot(errorMatch, snrs);
    ylabel("snr");
    xlabel("error");
    title("fb = 1/" + inter + "Tp");
    legend("Sign", "Matched");
end