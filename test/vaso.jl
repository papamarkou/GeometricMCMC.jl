# Create design matrix X and response variable y from vaso data array
vaso = readdlm("vaso.txt", ' ');
covariates = vaso[:, 1:end-1];
nData, nPars = size(covariates);

covariates = (bsxfun(-, covariates, mean(covariates, 1))
  ./repmat(std(covariates, 1), nData, 1));

polynomialOrder = 1;
X = ones(nData, nPars*polynomialOrder+1);
for i = 1:polynomialOrder
  X[:, ((i-1)*nPars+2):(i*nPars+1)] = covariates.^i;
end
nPars += 1

y = vaso[:, end];

# Create data dictionary
data = {"X"=>X, "y"=>y, "priorVar"=>100., "nData"=>nData};
