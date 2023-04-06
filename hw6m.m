clear, close all

% param
N = 10; % number of bits
Tp = 0.1; % half the pulse width
dt = Tp/50; % sampling frequency -- keep this constant
fb = 1/(2*Tp); % bit rate
dt_b = 1/fb; % bit period (how long between bits)
t = 0:dt:(dt_b*N); % time range -- not sure if this is right
% upper limit is the number of bit periods times the number of bits

%% PART A -- need graphs, etc
t_pulse = -Tp:dt:Tp;
p = 1-abs(t_pulse./Tp);
% creates triangular pulse that is 2*Tp wide in time

plot(t_pulse,p)


%% PART C
% testing: bits = 2*((rand(1,N)<0.5)-0.5);
bits = [repmat([1 1]',[5,1])]; % testing
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


