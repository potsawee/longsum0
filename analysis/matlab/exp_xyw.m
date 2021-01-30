clear
A = [
feature_xyw(512,144,128);
feature_xyw(512,144,256);
feature_xyw(512,144,512);
feature_xyw(512,144,64);
feature_xyw(512,144,32);
feature_xyw(1024,144,128);
feature_xyw(1024,144,256);
feature_xyw(1024,144,512);
feature_xyw(1024,144,64);
feature_xyw(1024,144,32);
feature_xyw(2048,144,128);
feature_xyw(2048,144,256);
feature_xyw(2048,144,512);
feature_xyw(2048,144,64);
feature_xyw(2048,144,32);
feature_xyw(3072,144,128);
feature_xyw(3072,144,256);
feature_xyw(3072,144,512);
feature_xyw(3072,144,64);
feature_xyw(3072,144,32);
feature_xyw(4096,144,128);
feature_xyw(4096,144,256);
feature_xyw(4096,144,512);
feature_xyw(4096,144,64);
feature_xyw(4096,144,32);
% -------------------------- %
feature_xyw(512,288,128);
feature_xyw(512,400,128);
feature_xyw(4096,288,32);
feature_xyw(4096,320,32);
feature_xyw(2048,320,64);
feature_xyw(2048,400,64);
feature_xyw(2048,288,128);
feature_xyw(2048,200,128);
feature_xyw(3072,200,64);
feature_xyw(1024,400,256);
feature_xyw(1024,360,512);
];

MEM = [
7.16;7.32;7.68;7.07;7.03;
7.94;8.28;8.97;7.79;7.71;
9.54;10.22;11.56;9.22;9.05;
11.17;12.17;14.16;10.68;10.44;
12.76;14.09;16.73;12.10;11.77;
% -------------------------- %
7.56;7.88;12.95;13.20;
10.11;10.55;10.29;9.82;
11.01;9.24;9.76
];

X_ls = (inv(transpose(A)*A)*transpose(A))*MEM

%%% RMSE %%%
MEM_ls = A*X_ls;
RMSE = sqrt(mean((MEM - MEM_ls).^2))

