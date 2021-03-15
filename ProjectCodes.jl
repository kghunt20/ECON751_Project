################################################################################
#0. Packages
################################################################################

using Random, Statistics, Distributions, Optim, DelimitedFiles, StatsBase

################################################################################
#1. Import data
################################################################################

cd("C:\\Users\\Kevin\\Documents\\Github\\ECON751_Project")
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
#s = schooling
#k = experience

#Calculate the deterministic part of the wage of the wife
function log_wage_det(parameters, s, k)

    r_0 = parameters[3]
    r_1 = parameters[4]
    r_2 = parameters[5]
    r_3 = parameters[6]

    r_0 + r_1*s + r_2*k

end

#Calculate the current period utility from working
function U1(parameters, c_y, w, s, k)

    c_y + w

end

#Calculate the current period utility from not working
function U0(parameters, c_y, w, s, k)

    α1 = parameters[7]
    α2 = parameters[8]

    c_y + α2*c_y + α1

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

    ξ_lb = quantile(Normal(0.0, σ_ξ), 0.0001)

    #Solve the model recursively
    for t = reverse(1:T), i_k = 1:t

        k = k_grid[i_k]

        #Deterministic part of wife's wage
        c_log_wage_det = log_wage_det(parameters, s, k)

        #Calculate ξ_star
        A = U0(parameters, y[t], c_log_wage_det, s, k) -
            U1(parameters, y[t], c_log_wage_det, s, k) +
            β*EV[t+1, i_k] - β*EV[t+1, i_k + 1]
        if A > 0
            ξ_star[t, i_k] = log(A) - c_log_wage_det
        else
            ξ_star[t, i_k] = ξ_lb
        end

        EV[t, i_k] = (y[t] + β*EV[t+1, i_k + 1] ) * (1-Φ(ξ_star[t,i_k]/σ_ξ)) +
                     exp(c_log_wage_det)*exp(0.5*σ_ξ^2)*
                        (1-Φ((ξ_star[t,i_k]-σ_ξ^2)/σ_ξ)) +
                    (U0(parameters, y[t], exp(c_log_wage_det), s, k) + β*EV[t+1,i_k])*
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

    LL = 0.0

    #Relevant parameters for likelihood function
    σ_u = (σ_ξ^2 + σ_η^2)^0.5
    ρ = σ_ξ/σ_u


    for id in unique(data[:,1])
        #Create subset of data for current wife
        data_now = data[data[:,1] .== id, :]

        data_y = data_now[:,8]
        data_wO = data_now[:,5]
        data_s = data_now[1,6]
        data_P = data_now[:,3]
        k_start = Int(data_now[1,4])
        data_k = data_now[:,4]

        ξ_star = get_ξ_star(parameters, data_y, data_s, k_start)

        i_k = 1

        for t = 1:T

            #Update LL if wife works.
            if data_P[t] == 1

                u = log(data_wO[t]) -
                    log_wage_det(parameters, data_s, data_k[t])
                A = (ξ_star[t, i_k] - ρ*σ_ξ/σ_u*u)/(σ_ξ*(1-ρ^2)^0.5)
                B = pdf(Normal(), u/σ_u)/σ_u
                LL -= log( (1.0-cdf(Normal(), A))*B)

                #Increase experience for next period
                i_k += 1

            #Update LL if wife doesn't work.
            elseif data_P[t] == 0

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
x0 =  [0.5987529235445073, 0.19021365175159843, 9.33874450407129,
       0.047565980905554126, 0.007830603645591645, -32.41353995455251,
       23912.216292202516, 0.008811804383663021]

#Start with only 250 couples of data to get a good initial guess quickly
#couples = sample(unique(store_data[:,1]), 100,  replace = false)
couples = sample(unique(data[:,1]), 250, replace = false)
data = original_data[subset.(original_data[:,1]),:]
res1 = optimize(likelihood, x0, iterations)

x0 = res1.minimizer

#Now use all the data
data = store_data
res1 = optimize(likelihood, x0)



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
