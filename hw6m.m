clear, close all

% param
N = 10; % number of bits
Tp = 0.1; % half the pulse width
dt = Tp/50; % sampling frequency -- keep this constant
fb = 1/(2*Tp); % bit rate
dt_b = 1/fb; % bit period (how long between bits)
t = 0:dt:(dt_b*N); % time range -- not sure if this is right
% upper limit is the number of bit periods times the number of bits


sigma = 1; % noise parameter for part e
%% PART A -- need graphs, etc
t_pulse = -Tp:dt:Tp;
p = 1-abs(t_pulse./Tp);
% creates triangular pulse that is 2*Tp wide in time

plot(t_pulse,p)


%% PART C
% testing: bits = 2*((rand(1,N)<0.5)-0.5);
bits = [ones([N,1])]; % testing signal of all ones length N (10)
figure,stem(bits)
% plot of some test bits we could use 

%% PART D




% this is where I got really confused. According to the bit rate the bits
% should be spaced 100 apart, and I think the first bit occurs at 51. So
% using this I created the following graph and it looks right, but then I
% do the convolution and its weird.
xn = zeros(N*dt_b/dt+1,1);

% select times where the bit period divides cleanly the time (this is where
% the bits should occur!)
bits_temp = bits;
for i = 51:(dt_b/dt):N*(dt_b/dt)+1
    xn(i) = bits_temp(1);
    bits_temp = bits_temp(2:end);
end
xn = xn';

figure, hold on
stem(xn)
plot(repmat(p(1:end-1),[1,10])) % 10 repeats of the triangular pulse. I cut
% off the last element in the pulse because otherwise it would be
% duplicating a zero

y = conv(xn,p);
% note: length of convolution vector is len(xn)+len(p)-1

% cropping out all the values in the convolution that are zero
y_crop = y(y ~= 0);
figure,plot(y_crop)
% if you zoom in really close to where the convolution touches x=0, you can
% see that it is flat for a little bit. This is where I am stuck I can't
% figure out why this is happening :(

% probably something to due with the overlapping of the triangles or
% something, maybe it could be fixed in part a



% Thomas
% How I think x should be generated: u take the bits defined in c, and put
% them into an array along with time. every Ts time, there will be one of
% the spikes corresponding to one in bits
% result is kinda same as urs evan, but maybe its more general? idk


% tx = t_pulse;

% makes a time vector of appropriate length. I chose Ts arbitrarily
Ts = 0.2;
tx = 0:dt:(N)*Ts;
xner = zeros(size(tx));
for i=0:N-1
    xner(abs(tx - i * Ts) < .0001) = bits(i+1);
end
y_conv = conv(xner, p);
% if length(xner) > length(p)
%     y_conv = conv(xner, p, 'same');
% else
%     y_conv = conv(p, xner, 'same');
% end


figure
hold on
plot([zeros(1, floor(length(p)/2)) xner zeros(1, floor(length(p)/2))] )
plot(y_conv)

%% part E
nt = sigma*randn(1,length(y_conv));
rt = nt + y_conv;
figure, plot(rt)
%% part F
tx2 = -Tp:dt:(N)*Ts + Tp;
xhat = zeros(size(rt));
xhater = zeros(size(rt));

pnegt = flip(p);
zn = conv(rt, pnegt, "same");

for i=0:N-1
    index = find(abs(tx2 - i* Ts) < .001);
    if rt(index) > 0
        xhat(index) = 1;
    else
        xhat(index) = -1;
    end


    if zn(index) > 0
        xhater(index) = 1;
    else
        xhater(index) = -1;
    end

end
figure, plot(xhat)


figure, plot(zn)
figure, plot(xhater)

Py = sum(y_conv.^2 * dt);
Pn = sum(nt.^2 * dt);
disp("SNR: " + Py/Pn)