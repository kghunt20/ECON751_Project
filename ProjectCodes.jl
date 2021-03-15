################################################################################
#0. Packages
################################################################################
# Hi
using Random, Statistics, Distributions, Optim, DelimitedFiles, StatsBase

################################################################################
#1. Import data
################################################################################

#Order of variables in data: id, age, lfp, x, wage, edu, lfp0, hinc
#I deleted variable names from this file, so it is just numers.
original_data = readdlm("data_age4554.txt");

################################################################################
#2. Define model structure (Utility functions and wage)
################################################################################

#σ_ξ= parameters[1]
#σ_η = parameters[2]
#r_0 = parameters[3]
#r_1 = parameters[4]
#r_2 = parameters[5]
#r_3 = parameters[6]
#α1 = parameters[7]
#α2 = parameters[8]
#s = years of school
#k = experience

#Calculate the deterministic part of the wage of the wife
function log_wage_det(parameters, s, k)

    r_0 = parameters[3]
    r_1 = parameters[4]
    r_2 = parameters[5]
    r_3 = parameters[6]

    r_0 + r_1*s + r_2*k + r_3*k^2

end

#Calculate the current period utility from working not including wife's income
#Total utility from working would be U1 + wife's wage
function U1(parameters, y, s, k)

    y

end

#Calculate the current period utility from not working
function U0(parameters, y, s, k)

    α1 = parameters[7]
    α2 = parameters[8]

    y + α2*y + α1

end

function Φ(x)
    cdf(Normal(), x)
end

function subset(x; couples = couples)
    x in couples
end

################################################################################
#3. Solve the model
################################################################################

#Solve the model for a given set of parameters and husbands income profile y,
#and education level for the wife s.
#T = Total number of periods the wife makes labor supply decisions.
#k_start = Starting experience of the wife.

function get_ξ_star(parameters, y, s, k_start; T=T, β = β)

    σ_ξ = parameters[1]

    #Generate a grid of possible experience the wife could achieve
    k_grid = k_start:(k_start + T-1)

    #Expected value function EV(t, k)
    EV = zeros(T+1, T+1)
    #Participation cut-off ϵ^* (ϵ_star)
    ξ_star = zeros(T, T)

    #A lower bound for ξ in the calculations below
    ξ_lb = quantile(Normal(0.0, σ_ξ), 0.0001)

    #Solve the model recursively
    for t = reverse(1:T), i_k = 1:t

        #Define current experience for the wife
        k = k_grid[i_k]

        #Deterministic part of wife's log wage. "c_" stands for current
        c_log_wage_det = log_wage_det(parameters, s, k)

        #Calculate ξ_star using equation on page 84 of Lecture 3
        A = U0(parameters, y[t], s, k) -
            U1(parameters, y[t], s, k) +
            β*EV[t+1, i_k] - β*EV[t+1, i_k + 1]
        if A > 0
            ξ_star[t, i_k] = log(A) - c_log_wage_det
        else
            ξ_star[t, i_k] = ξ_lb
        end

        #Calculate the expect value function (EV) using equation in lecture 3
        #page 87
        EV[t, i_k] = (y[t] + β*EV[t+1, i_k + 1] ) * (1-Φ(ξ_star[t,i_k]/σ_ξ)) +
                     exp(c_log_wage_det)*exp(0.5*σ_ξ^2)*
                        (1-Φ((ξ_star[t,i_k]-σ_ξ^2)/σ_ξ)) +
                    (U0(parameters, y[t], s, k) + β*EV[t+1,i_k])*
                        Φ(ξ_star[t, i_k]/σ_ξ) #Is there a typo in notes here?

    end

    #Deliverables
    ξ_star

end

################################################################################
#4. Likelihood Function
################################################################################

#This function calculates log-likelihood (LL) given our parameters and data
#T = Total number of periods the wife makes labor supply decisions.
#wO = Observed wage of wife (with measurement error)

function likelihood(parameters; data = data, T = T, β = β)

    σ_ξ = parameters[1]
    σ_η = parameters[2]

    #Initialize log-likelihood
    LL = 0.0

    #Relevant parameters for likelihood function
    σ_u = (σ_ξ^2 + σ_η^2)^0.5
    ρ = σ_ξ/σ_u

    #Loop over every couple in the data
    for id in unique(data[:,1])

        #Create subset of data for current wife
        data_now = data[data[:,1] .== id, :]

        #Unpack data_now
        yvec = data_now[:,8] #Husbands income
        wOvec = data_now[:,5] #Wife's observed income
        s = data_now[1,6] #Wife's schooling
        Pvec = data_now[:,3] #Wife's participation
        kvec = data_now[:,4] #Wife's experience

        ξ_star = get_ξ_star(parameters, yvec, s, kvec[1])

        i_k = 1

        #Loop over each time period
        for t = 1:T

            #Update LL if wife works using equaation on page 109 in lecture 3
            if Pvec[t] == 1

                u = log(wOvec[t]) -
                    log_wage_det(parameters, s, kvec[t])
                A = (ξ_star[t, i_k] - ρ*σ_ξ/σ_u*u)/(σ_ξ*(1-ρ^2)^0.5)
                B = pdf(Normal(), u/σ_u)/σ_u
                LL -= log( (1.0-cdf(Normal(), A))*B)

                i_k += 1

            #Update LL if wife doesn't work using equation on page 109 of lecture 3.
            elseif Pvec[t] == 0

                LL -= log(cdf(Normal(), ξ_star[t, i_k]/σ_ξ))

            end

        end

    end

    @show LL

end

################################################################################
#5. Estimate parameters by maximum likelihood
################################################################################

β = 0.95
#Number of periods the wife makes labor supply decisions.
T = 20

#Initial guess
x0 =  [0.5990950138078999, 0.1892819256699379, 9.295133566845593,
       0.046655500082713575, 0.014359272734607026, -0.00014647900745677907,
       24126.810548022222, 0.00978890544903218]

#Start with only 250 couples of data to get a good initial guess quickly
#couples = sample(unique(store_data[:,1]), 100,  replace = false)
couples = sample(unique(data[:,1]), 250, replace = false)
data = original_data[subset.(original_data[:,1]),:]
res1 = optimize(likelihood, x0)

#Update initial guess
x0 = res1.minimizer

#Now use all the data
data = original_data
res1 = optimize(likelihood, x0)
x0 = res1.minimizer


################################################################################
#6. Calculate standard errors (Optional)
################################################################################






################################################################################
#*******************************************************************************
################################################################################
#II.  Model Fit: In simulating data from your model, you should do 20
#     replications for each couple.  Then you should compare the simulations to
#     the actual data in terms of the following statistics:
################################################################################
#*******************************************************************************
################################################################################

function simulate_data(parameters, N, y, s, k_start; T = T, β = β)

    σ_ξ = parameters[1]
    σ_η = parameters[2]

    ξ_star = get_ξ_star(parameters, y, s, k_start)

    observations = zeros(T*N, 8)

    #Order of variables in data: id, age, lfp, x, wage, edu, lfp0, hinc
    i = 1
    for n = 1:N

        i_k = 1
        k = k_start
        wO = 0.0
        k_prime = k
        P = 0.0

        for t = 1:T

            ξ = randn()*σ_ξ

            if ξ > ξ_star[t, i_k]
                P = 1
                wO = exp(log_wage_det(parameters, s, k_start + i_k -1) +
                         randn()*σ_η)
                k_prime = k+1
                i_k = i_k + 1
            else
                P = 0
                wO = -9.0
            end

            observations[i,:] = [n, t, P, k, wO, s, 1, y[t]]
            k = k_prime
            i+=1

        end

    end

    observations

end

################################################################################
#1. The average number of period working over the 10 years overall and by
#   whether or not the woman has less than 12 years of schooling, exactly 12,
#   13-15 and 16+.
################################################################################







################################################################################
#2. The fraction of women working at each age.
################################################################################











################################################################################
#4. The fraction of women working by work experience levels (a) 10 years or
#   less, (b) 11-20, (c) 21+.
################################################################################










################################################################################
#5. The 2x2 table of labor force participation by lagged labor force
#   participation.
################################################################################




################################################################################
#*******************************************************************************
################################################################################
#III.  Counterfactual Analysis.
################################################################################
#*******************************************************************************
################################################################################

################################################################################
#1. Simulate data for participation and (accepted) wages for ages 55 to 64.
#   How many years, on average, will women work between those ages, overall and
#   by education?
################################################################################










################################################################################
#2. Assume that the government introduces a flat income tax on total earnings
#   (husband + wife) of 10 percent. Assume that couples report the woman’s
#   earnings accurately to the IRS, although the wage is misreported in our
#   data.  You should therefore use your estimate of the true wage for this
#   exercise.
#        (a) What will happen to the average number of years worked between
#            the ages 45-54?   What is the total revenue the IRS will collect?
#        (b)  Do the same for ages 55-64.
################################################################################























################################################################################
#3. Suppose the tax schedule is instead progressive.  The couple pays a 10
#   percent tax on the first 50,000 and a 20 percent tax on anything above that.
#   Assume that the couples reported the woman’s earnings accurately to the IRS,
#   although the wage is misreported in our data. You should therefore use your
#   estimate of the true wage for this exercise.a) What will happen to the
#   average number of years worked between the ages 45-54?   What is the total
#   revenue the IRS will collect? (b)  Do the same for ages 55-64.
################################################################################
