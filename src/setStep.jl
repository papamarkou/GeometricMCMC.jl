## Auxiliary functions for adjusting the drift or leap step in geometric MCMC 
## algorithms

# Generic function for step adjustment
function setStep(i::Int, acceptanceRatio::Float64, nMcmc::Int, nBurnin::Int, 
  currentStep::Float64, steps::Vector{Float64}, lowerRatio::Vector{Float64}, 
  upperRatio::Vector{Float64})
  if i == 1
    steps[1]
  elseif i <= 0.1*nBurnin
    if acceptanceRatio < lowerRatio[1]
      steps[2]
    elseif upperRatio[1] <= acceptanceRatio
      steps[3]
    else
      currentStep
    end
  elseif i <= 0.15*nBurnin
    if acceptanceRatio < lowerRatio[2]
      steps[3]
    elseif upperRatio[2] <= acceptanceRatio
      steps[4]
    else
      currentStep
    end
  elseif i <= 0.2*nBurnin
    if acceptanceRatio < lowerRatio[3]
      steps[4]
    elseif upperRatio[3] <= acceptanceRatio
      steps[5]
    else
      currentStep
    end
  elseif i <= 0.25*nBurnin
    if acceptanceRatio < lowerRatio[4]
      steps[5]
    elseif upperRatio[4] <= acceptanceRatio
      steps[6]
    else
      currentStep
    end
  elseif i <= 0.3*nBurnin
    if acceptanceRatio < lowerRatio[5]
      steps[6]
    elseif upperRatio[5] <= acceptanceRatio
      steps[7]
    else
      currentStep
    end
  elseif i <= 0.35*nBurnin
    if acceptanceRatio < lowerRatio[6]
      steps[7]
    elseif upperRatio[6] <= acceptanceRatio
      steps[8]
    else
      currentStep
    end
  elseif i <= 0.4*nBurnin
    if acceptanceRatio < lowerRatio[7]
      steps[8]
    elseif upperRatio[7] <= acceptanceRatio
      steps[9]
    else
      currentStep
    end
  elseif i <= 0.45*nBurnin
    if acceptanceRatio < lowerRatio[8]
      steps[9]
    elseif upperRatio[8] <= acceptanceRatio
      steps[10]
    else
      currentStep
    end
  elseif i <= 0.5*nBurnin
    if acceptanceRatio < lowerRatio[10]
      throw("Aborted: low acceptance ratio during burn-in")
    elseif acceptanceRatio < lowerRatio[9]
      steps[10]
    elseif upperRatio[9] <= acceptanceRatio
      steps[1]
    else
      currentStep
    end
  elseif i <= 0.7*nBurnin
    currentStep
  elseif i <= 0.75*nBurnin
    if acceptanceRatio < lowerRatio[10]
      throw("Aborted: low acceptance ratio during burn-in")
    else
      currentStep
    end
  elseif i <= 0.9*nBurnin
    currentStep
  elseif i <= nBurnin
    if acceptanceRatio < lowerRatio[10]
      throw("Aborted: low acceptance ratio during burn-in")
    elseif upperRatio[10] <= acceptanceRatio
      throw("Aborted: high acceptance ratio during burn-in")
    else
      currentStep
    end
  # 0.5*(nBurnin+nMcmc) =nBurnin+0.5*nPostBurnin
  elseif i <= 0.5*(nBurnin+nMcmc)
    currentStep
  elseif i <= nMcmc
    if acceptanceRatio < lowerRatio[10]
      throw("Aborted: low acceptance ratio during post burn-in")
    elseif upperRatio[10] <= acceptanceRatio
      throw("Aborted: high acceptance ratio during post burn-in")
    else
      currentStep
    end
  end
end

# Function for adjusting MALA drift step
malaLowerRatio = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.1]
malaUpperRatio = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.85, 0.8, 0.9]

setMalaDriftStep(i::Int, acceptanceRatio::Float64, nMcmc::Int, nBurnin::Int, 
  currentStep::Float64, steps::Vector{Float64}) =
  setStep(i::Int, acceptanceRatio::Float64, nMcmc::Int, nBurnin::Int, 
  currentStep::Float64, steps::Vector{Float64}, malaLowerRatio, malaUpperRatio)

# Function for adjusting SMMALA drift step
smmalaLowerRatio = copy(malaLowerRatio)
smmalaUpperRatio = copy(malaUpperRatio)

setSmmalaDriftStep(i::Int, acceptanceRatio::Float64, nMcmc::Int, nBurnin::Int, 
  currentStep::Float64, steps::Vector{Float64}) =
  setStep(i::Int, acceptanceRatio::Float64, nMcmc::Int, nBurnin::Int, 
  currentStep::Float64, steps::Vector{Float64}, smmalaLowerRatio,
  smmalaUpperRatio)

# Function for adjusting MMALA drift step
mmalaLowerRatio = copy(malaLowerRatio)
mmalaUpperRatio = copy(malaUpperRatio)

setMmalaDriftStep(i::Int, acceptanceRatio::Float64, nMcmc::Int, nBurnin::Int, 
  currentStep::Float64, steps::Vector{Float64}) =
  setStep(i::Int, acceptanceRatio::Float64, nMcmc::Int, nBurnin::Int, 
  currentStep::Float64, steps::Vector{Float64}, mmalaLowerRatio,
  mmalaUpperRatio)

# Function for adjusting HMC leap step
hmcLowerRatio = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.5, 0.1]
hmcUpperRatio = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.85, 0.8, 0.925]

setHmcLeapStep(i::Int, acceptanceRatio::Float64, nMcmc::Int, nBurnin::Int, 
  currentStep::Float64, steps::Vector{Float64}) = 
  setStep(i::Int, acceptanceRatio::Float64, nMcmc::Int, nBurnin::Int, 
  currentStep::Float64, steps::Vector{Float64}, hmcLowerRatio, hmcUpperRatio)

# Function for adjusting RMHMC leap step
rmhmcLowerRatio = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.1]
rmhmcUpperRatio = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.85, 0.8, 0.9]

setRmhmcLeapStep(i::Int, acceptanceRatio::Float64, nMcmc::Int, nBurnin::Int, 
  currentStep::Float64, steps::Vector{Float64}) = 
  setStep(i::Int, acceptanceRatio::Float64, nMcmc::Int, nBurnin::Int, 
  currentStep::Float64, steps::Vector{Float64}, rmhmcLowerRatio, 
  rmhmcUpperRatio)
