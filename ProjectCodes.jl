################################################################################
#0. Packages
################################################################################

using Random, Statistics, Distributions, Optim, DelimitedFiles

################################################################################
#1. Import data
################################################################################

cd("Documents/Github/ECON751_Project")
#Order of variables in data: id, age, lfp, x, wage, edu, lfp0, hinc
#I deleted variable names from this file, so it is just numers.
data = readdlm("data_age4554.txt");

################################################################################
#2. Define model structure (Utility functions and wage)
################################################################################

#σ_ϵ = parameters[1]
#σ_η = parameters[2]
#r_s = parameters[3]
#r_k = parameters[4]
#α1 = parameters[5]
#α0 = parameters[6]
#s = schooling of wife
#k = experience of wife

#Calculate the deterministic part of the wage of the wife
function log_wage_det(parameters, s, k)

    r_s = parameters[3]
    r_k = parameters[4]

    r_s*s + r_k*k

end

#Calculate the current period utility from working
function c_U1(parameters, c_y, w, s, k)

    c_y + w

end

#Calculate the current period utility from not working
function c_U0(parameters, c_y, w, s, k)

    α1 = parameters[5]
    α0 = parameters[6]

    c_y + α1*c_y + α0

end

################################################################################
#3. Solve the model
################################################################################

#Solve the model for a given set of parameters and husbands income profile y,
#and education level for the wife s.
#T = Total number of periods the wife makes labor supply decisions.
#k_start = Starting experience of the wife.
#Δϵ = grid size for random component of wages

function solve_model(parameters, y, s, k_start; T=T, β = β, Δϵ = Δϵ)

    σ_ϵ = parameters[1]

    #For the distribution of the wage shock ϵ, choose lower (lb) and upper (ub)
    #bounds for the grid of shocks.
    ϵ_lb, ϵ_ub = round.(quantile.(Normal(0.0, σ_ϵ), [0.0001, 0.9999]),
                        digits = Int(-log10(Δϵ)))

    #Generate ϵ_grid
    ϵ_grid = ϵ_lb:Δϵ:ϵ_ub
    N_ϵ = length(ϵ_grid)
    ϕ = pdf.(Normal(0.0, σ_ϵ), ϵ_grid)

    #Generate a grid of possible experience the wife could achieve
    k_grid = k_start:(k_start + T-1)

    #Intialize model functions:
    #Value function V(ϵ, t, k)
    V = zeros(N_ϵ, T, T)
    #Expected value function EV(t, k)
    EV = zeros(T+1, T+1)
    #Participation function P(ϵ, t, k)
    P = zeros(N_ϵ, T, T)
    #Participation cut-off ϵ^* (ϵ_start)
    ϵ_star = zeros(T, T)

    #Solve the model recursively
    for t = reverse(1:T), i_k = 1:t

        k = k_grid[i_k]

        #Deterministic part of wife's wage
        log_w_bar = log_wage_det(parameters, s, k)

        #Calculate ϵ_star, being careful to not take log of negative number :)
        A = c_U0(parameters, y[t], log_w_bar, s, k) - c_U1(parameters, y[t], log_w_bar, s, k) +
            β*EV[t+1, i_k] - β*EV[t+1, i_k + 1]
        if A > 0
            ϵ_star[t, i_k] = log(A) - log_w_bar
        else
            ϵ_star[t, i_k] = ϵ_lb
        end

        #Solve for value and participation function given the ϵ shock.
        for i_ϵ = 1:N_ϵ

            w = exp(log_wage_det(parameters, s, k) + ϵ_grid[i_ϵ])
            U1 = c_U1(parameters, y[t], w, s, k) + β*EV[t+1, i_k + 1]
            U0 = c_U0(parameters, y[t], w, s, k) + β*EV[t+1, i_k]

            P[i_ϵ, t, i_k] = (U1 > U0)
            V[i_ϵ, t, i_k] = maximum([U1 U0])

        end

        #Calculate expected value function using trapezoid rule.
        EV[t, i_k] = Δϵ*(sum(V[:, t, i_k] .* ϕ) - 0.5*(V[1,t,i_k]*ϕ[1] + V[N_ϵ,t,i_k]*ϕ[N_ϵ]))

    end

    #Deliverables
    ϵ_grid, P, ϵ_star

end

################################################################################
#4. Likelihood Function
################################################################################

#This function calculates log-likelihood (LL) given our parameters and data
#T = Total number of periods the wife makes labor supply decisions.
#wO = Observed wage of wife (with measurement error)
# !!! How will data be structured?

function likelihood(parameters; data = data, T = T, β = β)

    σ_ϵ = parameters[1]
    σ_η = parameters[2]
    r_s = parameters[3]
    r_k = parameters[4]
    α1 = parameters[5]
    α0 = parameters[6]

    LL = 0.0

    #Relevant parameters for likelihood function
    σ_u = (σ_ϵ^2 + σ_η^2)^0.5
    ρ = σ_ϵ/σ_u


    # !!! Parallelize this for sure
    for id in unique(data[:,1])
        #Create subset of data for current wife
        data_now = data[data[:,1] .== id, :]

        data_y = data_now[:,8]
        data_wO = data_now[:,5]
        data_s = data_now[1,6]
        data_P = data_now[:,3]
        k_start = Int(data_now[1,4])
        k_grid = k_start:(k_start + T - 1)


        ϵ_grid, P, ϵ_star = solve_model(parameters, data_y, data_s, k_start)

        i_k = 1

        for t = 1:T

            #Update LL if wife works.
            if data_P[t] == 1

                u = log(data_wO[t]) - log_wage_det(parameters, data_s, k_grid[i_k])
                A = (ϵ_star[t, i_k] - ρ*σ_ϵ/σ_u*u)/(σ_ϵ*(1-ρ^2)^0.5)
                B = pdf(Normal(), u/σ_u)/σ_u
                LL -= log( (1.0-cdf(Normal(), A))*B )

                #Increase experience for next period
                i_k += 1

            #Update LL if wife doesn't work.
            elseif data_P[t] == 0

                LL -= log(cdf(Normal(), ϵ_star[t, i_k]/σ_ϵ))

            end

        end

    end

    #Show the LL while optimize iterates so you know it is doing something
    @show LL

    LL

end

################################################################################
#5. Estimate parameters by maximum likelihood
################################################################################

β = 0.95
#Δϵ must be a multiple of 10.
Δϵ = 0.1
#Number of periods the wife makes labor supply decisions.
T = 20

#Initial guess
x0 = [1.0, 1.0, 0.1, 0.1, 0.1, 0.5]

#I currently set iterations = 100 so it deoesn't go on forever
@time optimize(likelihood, x0, iterations = 100)

################################################################################
#6. Calculate standard errors (Optional)
################################################################################






################################################################################
#*******************************************************************************
################################################################################
#II.  Model Fit
################################################################################
#*******************************************************************************
################################################################################

################################################################################
#1. Simulate 20 wives for each real wife in the data.
################################################################################


################################################################################
#2. Function that calculates average number of periods worked over lifetime for
#   subsamples in both data and simulations:
#   i. Everyone
#   ii. Less than 12 years education
#   iii. 12 years eduation
#   iv. 13-15 years education
#   v. 16+ years education
################################################################################

################################################################################
#3. Function that calculates fraction of women working at each age in both data
#   and simulations.
################################################################################

################################################################################
#4. ....
################################################################################




################################################################################
#*******************************************************************************
################################################################################
#III.  Counterfactual Analysis.
################################################################################
#*******************************************************************************
################################################################################
