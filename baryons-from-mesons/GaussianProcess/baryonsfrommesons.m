% A quick way to get this script running is to first download the GPML
% package for Matlab at http://www.gaussianprocess.org/gpml/code/matlab/doc/ 
% (written by Carl Edward Rasmussen and Hannes Nickisch, based on previous versions
% written by Carl Edward Rasmussen and Chris Williams). We employ version
% 4.2. 
%
% Ensure the string variable 'gpmlpath' below stores the right path to the GPML package.  
% Store in 'direc' the path to the current script. Run the script.

gpmlpath='~/Downloads/gpml-matlab-v4.2-2018-06-11';% path to the GPML package
addpath(genpath(gpmlpath));

scriptdirec='~/Downloads/Baryons-from-Mesons-ML-master/GaussianProcess';% path to the current script

produceplots=0; % choose 1 to create plots
mesoninputs=readmatrix([scriptdirec,'/data/mesoninputs.dat']);
mesonmasses=readmatrix([scriptdirec,'/data/mMass.dat']);

baryoninputs=readmatrix([scriptdirec,'/data/baryoninputs.dat']);
baryonmasses=readmatrix([scriptdirec,'/data/bMass.dat']);

exinputs=readmatrix([scriptdirec,'/data/exinputs.dat']);
exmasses=readmatrix([scriptdirec,'/data/eMass.dat']);

unitinputs=readmatrix([scriptdirec,'/data/unitinputs.dat']);


dataI = mesoninputs;% train inputs with an additional feature that ranks mesons by masses
dataO = mesonmasses;% train outputs
testI = baryoninputs;% test inputs
testO = baryonmasses;% test outputs


numinputfeatures = length(dataI(1,:));
dataII=dataI;
dataII(:,numinputfeatures) = zeros(length(dataI),1);% "physical" train inputs: train inputs with additional feature set to 0. 



% massaging data
normalisemasses=1;

if normalisemasses==1
    dataO=normalize(dataO);
end
mu=mean(mesonmasses);sigma=std(mesonmasses);
%% GP set up
list = {{@covRQard}};%

meanfunc=[];hyp.mean=[];% mean function
covfunc = {@covSum, list};% covariance function

c=['@(D)',feval(covfunc{:})];
fh=str2func(c);
disp('# hyperparameters: ');
disp(fh(numinputfeatures));

a1=0;b1=0;
hyp.cov = (a1+(b1-a1)*rand(1,fh(numinputfeatures)));
disp(hyp.cov);

likfunc = @likGauss;% likelihood 
sn = 1.e-6;
hyp.lik = log(sn);

inf = @infGaussLik; % inference method

%% ML optimisation
maxiters = 8000;

hyp2 = minimize(hyp, @gp, -maxiters, inf, meanfunc, covfunc, likfunc, dataI, dataO);
[nlml, dnlml] = gp(hyp2, inf, meanfunc, covfunc, likfunc, dataI, dataO); %training
[trainmean, trainvars, trainfmu, trainfs2] = gp(hyp2, inf, meanfunc, covfunc, likfunc, dataI, dataO, dataI); % testing on training data
[trainmeanphys, trainvarsphys, trainfmuphys, trainfs2phys] = gp(hyp2, inf, meanfunc, covfunc, likfunc, dataI, dataO, dataII); % testing on (physical) training data
[testmean, testvars, testfmu, testfs2] = gp(hyp2, inf, meanfunc, covfunc, likfunc, dataI, dataO, testI); % testing on test data
[unitmean, unitvars, unitfmu, unitfs2] = gp(hyp2, inf, meanfunc, covfunc, likfunc, dataI, dataO, unitinputs); % testing on unit vectors
[exmean, exvars, exfmu, exfs2] = gp(hyp2, inf, meanfunc, covfunc, likfunc, dataI, dataO, exinputs); % testing on exotics

if normalisemasses==1 %mu=mean(mesonmasses);sigma=std(mesonmasses);
    dataO     = unnormalize(dataO,mu,sigma);
    trainmean = unnormalize(trainmean,mu,sigma);
    trainmeanphys = unnormalize(trainmeanphys,mu,sigma);
    testmean  = unnormalize(testmean,mu,sigma);
    unitmean  = unnormalize(unitmean,mu,sigma);
    exmean    = unnormalize(exmean,mu,sigma);
end

%% display results
disp('NEGATIVE LOG MARGINAL LIKELIHOOD: ');
disp(nlml);
disp(['inferred noise standard deviation ~ vs ~ ','actual noise standard deviation']);
disp([exp(hyp2.lik), exp(hyp.lik)]);

disp('characteristic length scales');
disp(exp(hyp2.cov'));
disp(newline);
disp (['''NATURAL SCALE'' -> MeV, ''DATA SCALE'' -> log MeV',newline,newline]);
disp('AVERAGE GP ERRORS in NATURAL SCALE:');
disp ('proton, neutron')
disp ([naturalerrors(testO(1),testmean(1),testvars(1)),naturalerrors(testO(2),testmean(2),testvars(2))]);
disp ('mesons (unphysical inputs), mesons (physical inputs), baryons, exotics ')
disp ([naturalerrors(dataO(4:length(dataO))',trainmean(4:length(dataO))',trainvars(4:length(dataO))'), naturalerrors(dataO(4:length(dataO))',trainmeanphys(4:length(dataO))',trainvarsphys(4:length(dataO))'),naturalerrors(testO',testmean',testvars'),naturalerrors(exmasses',exmean',exvars')]);
disp(newline)
disp('AVERAGE GP ERRORS in DATA SCALE:');
disp ('proton, neutron')
disp ([logerrors(testO(1),testmean(1)),logerrors(testO(2),testmean(2))]);
disp ('mesons (unphysical inputs), mesons (physical inputs), baryons, exotics ')
disp ([logerrors(dataO(4:length(dataO))',trainmean(4:length(dataO))'),logerrors(dataO(4:length(dataO))',trainmeanphys(4:length(dataO))'),logerrors(testO',testmean'),logerrors(exmasses',exmean')]);

%% store GP outputs (means and variances)
writematrix(trainmean,[scriptdirec,'/results/trainmean.txt']);
writematrix(trainvars,[scriptdirec,'/results/trainvars.txt']);

writematrix(trainmeanphys,[scriptdirec,'/results/trainmeanphys.txt']);
writematrix(trainvarsphys,[scriptdirec,'/results/trainvarsphys.txt']);

writematrix(testmean,[scriptdirec,'/results/testmean.txt']);
writematrix(testvars,[scriptdirec,'/results/testvars.txt']);

writematrix(exmean,[scriptdirec,'/results/exmean.txt']);
writematrix(exvars,[scriptdirec,'/results/exvars.txt']);

writematrix(unitmean,[scriptdirec,'/results/unitmean.txt']);
writematrix(unitvars,[scriptdirec,'/results/unitvars.txt']);

%% plots
if produceplots==1
    xs = linspace(4.5, 9.5, 60)';ys = xs;
    f1= figure(1);
    clf(f1);
    hold on; 
    errorbar((exmasses),(exmean),sqrt(exvars),'o','MarkerSize',3,...
        'MarkerEdgeColor','red','MarkerFaceColor','red');
    plot(xs,ys); 
    plot(xs,xs+log(8/9),'blue');
    plot(xs,xs+log(10/9),'blue');
    hold off;

    f2= figure(2);
    clf(f2);
    hold on; 
    errorbar((testO),(testmean),sqrt(testvars),'o','MarkerSize',3,...
        'MarkerEdgeColor','red','MarkerFaceColor','red');
    plot(xs,ys); 
    plot(xs,xs+log(8/9),'blue');
    plot(xs,xs+log(10/9),'blue');
    hold off;

    f3= figure(3);
    clf(f3);
    hold on; 
    errorbar((dataO),(trainmean),sqrt(trainvars),'o','MarkerSize',3,...
         'MarkerEdgeColor','blue','MarkerFaceColor','blue');
    plot(xs,ys); 
    plot(xs,xs+log(8/9),'blue');
    plot(xs,xs+log(10/9),'blue');
    hold off;

    f4= figure(4);
    clf(f4);
    hold on; 
    errorbar((dataO),(trainmeanphys),sqrt(trainvarsphys),'o','MarkerSize',3,...
         'MarkerEdgeColor','blue','MarkerFaceColor','blue');
    plot(xs,ys); 
    plot(xs,xs+log(8/9),'blue');
    plot(xs,xs+log(10/9),'blue');
    hold off;
end
%% functions
function err = naturalerrors(v1,v2,varsV2)% v1: vector of actual values, v2: measured
    v1=exp(v1);
    v2=exp(v2+(1/2)*varsV2);
    err=100*sum(abs(rdivide(v1-v2,v1)))/length(v1);
end

function err = logerrors(v1,v2)% v1: vector of actual values, v2: measured
    err=100*sum(abs(rdivide(v1-v2,v1)))/length(v1);
end

function vals1 = normalize(vals)
    vals1=(vals-mean(vals))/std(vals);
end

function vals = unnormalize(vals1,mu,sigma)
         vals= vals1*sigma + mu;
end