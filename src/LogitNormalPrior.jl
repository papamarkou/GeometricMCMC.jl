function logPrior(pars::Vector{Float64}, data::Dict{Any, Any})
  return (-dot(pars,pars)/data["priorVar"]
    -data["nPars"]*log(2*pi*data["priorVar"]))/2
end

function = logLikelihood(pars::Vector{Float64}, data::Dict{Any, Any})
  XPars = pars*data["X"]
  return data["y"]*XPars
  Xb = obj.X*b;
  return Xb'*obj.y-sum(log(1+exp(Xb)));
end

classdef LogitNormalPrior
% LogitNormalPrior: Class for Bayesian logit model with a Normal prior
% N(0, aI).

% Written by Theodore Papamarkou.
% (c) UCL, Department of Statistical Science. All rights reserved.

  properties (SetAccess=private)
    X % Design matrix
    y % Response variables
    a % Prior variance
    nx % Number of rows of X
    np % Number of columns of X
  end
  methods
    function obj = LogitNormalPrior(X, y, a)
      obj.X = X;
      obj.y = y;
      obj.a = a;
      [obj.nx, obj.np] = size(X);
    end
    function output = logPrior(obj, b)
      output = sum(-b.^2/(2*obj.a)-log(2*pi*obj.a)/2);
    end
    function output = logLikelihood(obj, b)
      Xb = obj.X*b;
      output = Xb'*obj.y-sum(log(1+exp(Xb)));
    end
    function output = logPosterior(obj, b)
      output = logLikelihood(obj, b)+logPrior(obj, b);
    end    
    function output = gradLogPosterior(obj, b)
      output = obj.X'*(obj.y-(1./(1+exp(-obj.X*b)))) ...
        -eye(obj.np)*(1/obj.a)*b;
    end
    function output = metricTensor(obj, b)
      p = 1./(1+exp(-obj.X*b));
      t1 = p.*(ones(obj.nx, 1)-p);
      output = (obj.X'.*repmat(t1', obj.np, 1))*obj.X ...
        +(eye(obj.np)./obj.a);
    end
    function output = derivativeMetricTensor(obj, b)
      p = 1./(1+exp(-obj.X*b));
      v = p.*(ones(obj.nx, 1)-p);
      T1 = zeros(obj.nx, obj.np);
      output = cell(1, obj.np);
      for i = 1:(obj.np)
        for j =1:obj.np
          T1(:, j) = (obj.X(:, j).*(v.*((1-2*p).*obj.X(:, i))));
        end
        output{i} = T1'*(obj.X);
      end
    end
    function output = samplePrior(obj)
      output = sqrt(obj.a)*randn(obj.np, 1);
    end
  end
end
