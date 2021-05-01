clc; clear; close all;
%% data1 
load fisheriris
X = meas;
y = species;
%% data2 
% load ionosphere % Contains X and y variables

N = size(X,1);
M= size(X,2);
%% Visualize data
figure;
gscatter(X(:,1), X(:,2), y,'rgb','osd');
xlabel('Sepal length');
ylabel('Sepal width');
%% Split the datasets randomly into a training (80 %) and a test set (20 %)
rng('default'); %be same index
PD = 0.80 ;  % percentage 80

%train
rng(1)% For reproducibility
idx = randperm(N);
X_train = X(idx(1:round(N*PD)),:); 
y_train = y(idx(1:round(N*PD)),:); 
%test
rng(1)% For reproducibility
X_test = X(idx(round(N*PD)+1:end),:);
y_test = y(idx(round(N*PD)+1:end),:);

%% Using linear SVM

Mdl  = fitcecoc(X_train,y_train);

labels = predict(Mdl,X_test);

%% decision tree

% Mdl = fitctree(X_train,y_train);
% 
% labels = predict(Mdl,X_test);

%% knn
% 
% Mdl = fitcknn(X_train,y_train);
% 
% labels = predict(Mdl,X_test);

%% confusion matris
% % [C,order] = confusionmat(y_test,labels);
figure;
cm = confusionchart(y_test,labels); title('confusion matris');
xlabel('predicted class'); ylabel('true class');