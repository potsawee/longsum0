function [out] = feature_xyw(x,y,w)
%FEATURE Summary of this function goes here
%   Detailed explanation goes here

% scale=[1,1000,100,100,100000,100000,10000]

out1=1  / 1; 
out2=x  / 1000;
out3=y  / 100;
out4=w  / 100;
out5=x*y / 100000;
out6=x*w / 100000;
out7=y^2 / 10000;

% out = [out1,out2,out4,out6];
out = [out1,out2,out3,out4,out5,out6,out7];

end