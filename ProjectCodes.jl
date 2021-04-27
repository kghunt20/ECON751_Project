################################################################################
#0. Packages
################################################################################
# Hi
using Random, Statistics, Distributions, Optim, DelimitedFiles, StatsBase, ForwardDiff, DataFrames, LinearAlgebra

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
#α3 = parameters[9]
#α4 = parameters[10]
#α5 = parameters[11]
#s = years of school
#k = experience

#Calculate the deterministic part of the wage of the wife
function log_wage_det(parameters::AbstractVector{Z}, s, k, a, y) where Z

    r_0 = parameters[3]
    r_1 = parameters[4]
    r_2 = parameters[5]
    r_3 = parameters[6]

    r_0 + r_1*s + r_2*k + r_3*k^2

end

#Calculate the current period utility from working not including wife's income
#Total utility from working would be U1 + wife's wage
function U1(parameters::AbstractVector{Z}, y, prev_P, s, k; τ1 = 0.0, τ2 = 0.0, L = 50000) where Z

    if y < L
        (1-τ1) * y
    else
        (1-τ1)*L + (1-τ2)* (y-L)
    end

end

#Calculate the current period utility from not working
function U0(parameters::AbstractVector{Z}, y, prev_P, s, k; τ1 = 0.0, τ2 = 0.0, L = 50000) where Z

    α1 = parameters[7]
    α2 = parameters[8]
    α3 = parameters[9]
    α4 = parameters[10]
    α5 = parameters[11]

    if y < L
        ((1-τ1)*y)*(1+α2) + α1 + α3*s + α4*k + α5*(prev_P-1)
    else
        ((1-τ1)*L + (1-τ2)*(y - L))*(1+α2) + α1 + α3*s + α4*k + α5*(prev_P-1)
    end

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

function get_ξ_star(
    parameters::AbstractVector{Z},
    y,
    s,
    k_start;
    T = T,
    β = β,
    τ1 = 0.0,
    τ2 = 0.0,
    L = 50000,
) where Z

    σ_ξ = parameters[1]

    #Generate a grid of possible experience the wife could achieve
    k_grid = k_start:(k_start+T-1)

    #Expected value function EV(t, k)
    EV = zeros(Z, T + 1, T + 1, 2)
    #Participation cut-off ϵ^* (ϵ_star)
    ξ_star = zeros(Z, T, T, 2)

    #A lower bound for ξ in the calculations below
    ξ_lb = quantile(Normal(0.0, σ_ξ), 0.0001)

    #Solve the model recursively
    for t in reverse(1:T), i_k = 1:t, prev_P = 1:2

        #Define current experience for the wife
        k = k_grid[i_k]

        #Deterministic part of wife's log wage. "c_" stands for current
        c_log_wage_det = log_wage_det(parameters, s, k, t + 44, y[t])

        #Calculate ξ_star using equation on page 84 of Lecture 3

        #The cut-off for the wife where the higher tax kicks in.
        L_wife = maximum([0.0, L - y[t]])

        A =
            (
                U0(parameters, y[t], prev_P, s, k; τ1 = τ1, τ2 = τ2, L = L) -
                U1(parameters, y[t], prev_P, s, k; τ1 = τ1, τ2 = τ2, L = L) +
                β * EV[t+1, i_k, prev_P] - β * EV[t+1, i_k+1, prev_P]
            ) / (1 - τ1)
        if A > 0
            ξ_star[t, i_k, prev_P] = log(A) - c_log_wage_det
        else
            ξ_star[t, i_k, prev_P] = ξ_lb
        end

        #If the calculation before put the wife above the cut-off, redo calculation
        if exp(c_log_wage_det + ξ_star[t, i_k, prev_P]) > L_wife

            A =
                (
                    U0(parameters, y[t], prev_P, s, k; τ1 = τ1, τ2 = τ2, L = L) -
                    U1(parameters, y[t], prev_P, s, k; τ1 = τ1, τ2 = τ2, L = L) +
                    β * EV[t+1, i_k, prev_P] - β * EV[t+1, i_k+1, prev_P] - (1 - τ1) * L_wife
                ) / (1 - τ2) + L_wife

            if A > 0
                ξ_star[t, i_k, prev_P] = log(A) - c_log_wage_det
            else
                ξ_star[t, i_k, prev_P] = ξ_lb
            end

        end

        #Calculate the expect value function (EV) using equation in lecture 3
        #page 87

        w = exp(c_log_wage_det)
        ξ_tax2 = ifelse(L_wife > 0, log(L_wife) - c_log_wage_det, ξ_lb)
        ξ_tax2 = maximum([ξ_tax2, ξ_star[t, i_k, prev_P]])


        D1 = U1(parameters, y[t], prev_P, s, k; τ1 = τ1, τ2 = τ2, L = L) + β*EV[t+1, i_k+1, prev_P]
        P1 = (1 - Φ(ξ_star[t, i_k, prev_P] / σ_ξ))

        D2 = (1-τ1)*w
        P2 = exp(0.5 * σ_ξ^2)*(1 - Φ((ξ_star[t, i_k, prev_P] - σ_ξ^2) / σ_ξ))

        D3 = (τ2 - τ1)*w
        P3 = exp(0.5 * σ_ξ^2)*(1 - Φ((ξ_tax2 - σ_ξ^2) / σ_ξ))

        D4 = (τ2 - τ1)*L_wife
        P4 = (1 - Φ(ξ_tax2 / σ_ξ))

        D5 = U0(parameters, y[t], prev_P, s, k; τ1 = τ1, τ2 = τ2, L = L) + β*EV[t+1, i_k, prev_P]
        P5 = Φ(ξ_star[t, i_k, prev_P] / σ_ξ)



        EV[t, i_k, prev_P] = D1*P1 +D2*P2 - D3*P3 + D4*P4 +D5*P5

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

function likelihood(
    parameters::AbstractVector{Z};
    data = data,
    T = T,
    β = β,
    τ1 = 0.0,
    τ2 = 0.0,
    L = 50000,
) where Z

    σ_ξ = parameters[1]
    σ_η = parameters[2]

    #Initialize log-likelihood
    LL = 0.0

    #Relevant parameters for likelihood function
    σ_u = (σ_ξ^2 + σ_η^2)^0.5
    ρ = σ_ξ / σ_u

    #Loop over every couple in the data
    for id in unique(data[:, 1])

        #Create subset of data for current wife
        data_now = data[data[:, 1].==id, :]

        #Unpack data_now
        yvec = data_now[:, 8] #Husbands income
        wOvec = data_now[:, 5] #Wife's observed income
        s = data_now[1, 6] #Wife's schooling
        Pvec = data_now[:, 3] #Wife's participation
        kvec = data_now[:, 4] #Wife's experience
        Prev_Pvec = zeros(Int, 10)
        Prev_Pvec[1] = data_now[1,7] + 1
        Prev_Pvec[2:10] = data_now[1:9, 3] .+ 1


        ξ_star =
            get_ξ_star(parameters, yvec, s, kvec[1]; τ1 = τ1, τ2 = τ2, L = L)

        i_k = 1

        #Loop over each time period
        for t = 1:T

            #Update LL if wife works using equaation on page 109 in lecture 3
            if Pvec[t] == 1

                u =
                    log(wOvec[t]) -
                    log_wage_det(parameters, s, kvec[t], t + 44, yvec[t])
                A = (ξ_star[t, i_k, Prev_Pvec[t]] - ρ * σ_ξ / σ_u * u) / (σ_ξ * (1 - ρ^2)^0.5)
                B = pdf(Normal(), u / σ_u) / σ_u
                LL -= log((1.0 - cdf(Normal(), A)) * B)

                i_k += 1

                #Update LL if wife doesn't work using equation on page 109 of lecture 3.
            elseif Pvec[t] == 0

                LL -= log(cdf(Normal(), ξ_star[t, i_k, Prev_Pvec[t]] / σ_ξ))

            end

        end

    end

    LL

end


################################################################################
#5. Estimate parameters by maximum likelihood
################################################################################


β = 0.95
#Number of periods the wife makes labor supply decisions.
T = 20

#Old Initial guess without experience in the utility.
x0 = [
    0.5990950138078999,
    0.1892819256699379,
    9.295133566845593,
    0.046655500082713575,
    0.014359272734607026,
    -0.00014647900745677907,
    24126.810548022222,
    0.00978890544903218,
    0.0,
]

#Initial guess with experience, school, and previous work decision in utility.

x0 = [0.5999902773217562,
    0.1830464849167745,
    9.380314493947242,
    0.04007368511420846,
    0.015969564714597234,
    -0.00024363065608372497,
    27283.44998663911,
    0.014812008148388267,
    -184.64192445511404,
    -29.799206467841824,
    -2080.5556691604734
]




#Start with only 250 couples of data to get a good initial guess quickly
#couples = sample(unique(store_data[:,1]), 100,  replace = false)
data = original_data
couples = sample(unique(data[:, 1]), 250, replace = false)
data = original_data[subset.(original_data[:, 1]), :]
res1 = optimize(likelihood, x0, iterations = 5000)

#Update initial guess
x0 = res1.minimizer

#Now use all the data
data = original_data
res2 = optimize(likelihood, x0, iterations = 5000)
xhat = res2.minimizer

@show res2
@show res2.minimizer


################################################################################
#6. Calculate standard errors (Optional)
################################################################################
k = size(xhat)[1]
omega = zeros(k,k)
for id in unique(data[:, 1])
    g = ForwardDiff.gradient(x->likelihood(x,data=data[data[:,1].==id,:]), xhat)
    global omega += g * g'
end
h = ForwardDiff.hessian(likelihood, xhat)

Avar_covar = inv(h) * omega * inv(h)
Avar = diag(Avar_covar)
@show xhat
@show Avar


################################################################################
#*******************************************************************************
################################################################################
#II.  Model Fit: In simulating data from your model, you should do 20
#     replications for each couple.  Then you should compare the simulations to
#     the actual data in terms of the following statistics:
################################################################################
#*******************************************************************************
################################################################################


function simulate_obs(parameters, N, y, s, k_start, prev_P;
                      T = T, β = β, τ1 = 0.0, τ2 = 0.0, L = 50000
                      )

    σ_ξ = parameters[1]
    σ_η = parameters[2]

    ξ_star = get_ξ_star(parameters, y, s, k_start; τ1 = τ1, )

    observations = zeros(T * N, 8)


    i = 1
    for n = 1:N
        i_k = 1
        k = k_start
        wO = 0.0
        k_prime = k
        P = 0.0

        for t = 1:T
            ξ = randn() * σ_ξ

            if ξ > ξ_star[t, i_k, prev_P]
                P = 1
                wO = exp(
                    log_wage_det(
                        parameters,
                        s,
                        k_start + i_k - 1,
                        t + 44,
                        y[t],
                    ) +
                    randn() * σ_η +
                    ξ,
                )
                k_prime = k + 1
                i_k = i_k + 1
                prev_P = 2
            else
                P = 0
                wO = -9.0
                prev_P = 1
            end

            observations[i, :] = [44 + t, P, k, wO, s, 1, y[t], n]
            k = k_prime
            i += 1

        end

    end

    observations

end


function get_simulated_data(x0, N;
                            β = β, T = T, data = original_data,
                            τ1 = 0.0, τ2 = 0.0, L = 50000)

    simulated_data = zeros(T * N * length(unique(data[:, 1])), 9)

    start = 1
    stop = N * T

    for id in unique(data[:, 1])

        #Order of variables in data: id, age, lfp, x, wage, edu, lfp0, hinc,
        #rep_id

        yvec = data[data[:, 1].==id, 8]
        s = data[data[:, 1].==id, 6][1]
        k_start = data[data[:, 1].==id, 4][1]
        prev_P = Int(data[data[:,1].==id,7][1]) + 1

        simulated_data[start:stop, 1] .= id
        simulated_data[start:stop, 2:end] =
            simulate_obs(x0, N, yvec, s, k_start, prev_P; τ1 = τ1, τ2 = τ2, L = L)

        start += N * T
        stop += N * T

    end

    simulated_data

end


data = original_data

simulated_data = get_simulated_data(xhat, 20)


#outfile = "simulated_data.txt"
#f = open(outfile, "w")
#for i = 1:size(simulated_data, 1)
#    println(f, simulated_data[i, :])
#end


################################################################################
#1. The average number of period working over the 10 years overall and by
#   whether or not the woman has less than 12 years of schooling, exactly 12,
#   13-15 and 16+.
################################################################################


simulated_avg10 = zeros(length(1:5))
original_avg10 = zeros(length(1:5))

function overall_lfp(smin, smax, data)
    temp = data[(data[:, 6].>=smin).&(data[:, 6].<=smax).&(data[:, 2].<=54), :]
    df = convert(DataFrame, temp)
    gdf = groupby(df, [:x1, :x9])
    ss = combine(gdf, :x3 => sum)
    avg = mean(ss[:, 3])
    avg
end

data = simulated_data
simulated_avg10[1] = overall_lfp(0, 11, data)
simulated_avg10[2] = overall_lfp(12, 12, data)
simulated_avg10[3] = overall_lfp(13, 15, data)
simulated_avg10[4] = overall_lfp(16, 54, data)
simulated_avg10[5] = overall_lfp(0, 54, data)

data = original_data
data = hcat(data, zeros(length(data[:, 1]))) # fake rep_id
original_avg10[1] = overall_lfp(0, 11, data)
original_avg10[2] = overall_lfp(12, 12, data)
original_avg10[3] = overall_lfp(13, 15, data)
original_avg10[4] = overall_lfp(16, 54, data)
original_avg10[5] = overall_lfp(0, 54, data)

simulated_avg10
original_avg10

################################################################################
#2. The fraction of women working at each age.
################################################################################
ages = 45:64

simulated_LFP = zeros(length(ages))
original_LFP = zeros(length(ages))

for i = 1:length(ages)
    simulated_LFP[i] =
        mean((simulated_data[simulated_data[:, 2].==ages[i], 3] .== 1.0))
    original_LFP[i] =
        mean((original_data[original_data[:, 2].==ages[i], 3] .== 1.0))

end

using Plots

plot(
    ages,
    simulated_LFP,
    label = "Simulated Data",
    lw = 3,
    ylim = (0, 1),
    main = "LFP",
)
plot!(ages, original_LFP, label = "Original Data", lw = 3)
title!("LFP")

################################################################################
#3. The fraction of women working by work experience levels (a) 10 years or
#   less, (b) 11-20, (c) 21+.
################################################################################

simulated_byK = zeros(length(1:3))
original_byK = zeros(length(1:3))

function lfp_byK(kmin, kmax, data)
    temp = data[(data[:, 4].>=kmin).&(data[:, 4].<=kmax).&(data[:, 2].<=54), :]
    df = convert(DataFrame, temp)
    gdf = groupby(df, [:x1, :x9])
    ss = combine(gdf, :x3 => mean)
    avg = mean(ss[:, 3])
    avg
end

data = simulated_data
simulated_byK[1] = lfp_byK(0, 10, data)
simulated_byK[2] = lfp_byK(11, 20, data)
simulated_byK[3] = lfp_byK(21, 54, data)

data = original_data
data = hcat(data, zeros(length(data[:, 1]))) # fake rep_id
original_byK[1] = lfp_byK(0, 10, data)
original_byK[2] = lfp_byK(11, 20, data)
original_byK[3] = lfp_byK(21, 54, data)

################################################################################
#4. The mean wage of working women overall, by the four education classes in #1
#   above and by age.
################################################################################

### overall ###
temp = simulated_data[(simulated_data[:, 3].==1), :]
df = convert(DataFrame, temp)
gdf = groupby(df, [:x1, :x9])
ss = combine(gdf, :x5 => mean)
simulated_wage = mean(ss[:, 3])
original_wage = mean(original_data[original_data[:, 3].==1, 5])



### by education class ###
simulated_wage_byedu = zeros(length(1:4))
original_wage_byedu = zeros(length(1:4))

function wage_byedu(smin, smax, data)
    temp = data[
        (data[
            :,
            6,
        ].>=smin).&(data[:, 6].<=smax).&(data[:, 2].<=54).&(data[:, 3].==1),
        :,
    ]
    df = convert(DataFrame, temp)
    gdf = groupby(df, [:x1, :x9])
    ss = combine(gdf, :x5 => mean)
    avg = mean(ss[:, 3])
    avg
end

data = simulated_data
simulated_wage_byedu[1] = wage_byedu(0, 11, data)
simulated_wage_byedu[2] = wage_byedu(12, 12, data)
simulated_wage_byedu[3] = wage_byedu(13, 15, data)
simulated_wage_byedu[4] = wage_byedu(16, 54, data)

data = original_data
data = hcat(data, zeros(length(data[:, 1]))) # fake rep_id
original_wage_byedu[1] = wage_byedu(0, 11, data)
original_wage_byedu[2] = wage_byedu(12, 12, data)
original_wage_byedu[3] = wage_byedu(13, 15, data)
original_wage_byedu[4] = wage_byedu(16, 54, data)

### by age ###
ages = 45:54
simulated_wage_byage = zeros(length(ages))
original_wage_byage = zeros(length(ages))

for i = 1:length(ages)
    temp = simulated_data[
        (simulated_data[:, 2].==ages[i]).&(simulated_data[:, 3].==1),
        :,
    ]
    df = convert(DataFrame, temp)
    gdf = groupby(df, [:x1, :x9])
    ss = combine(gdf, :x5 => mean)
    simulated_wage_byage[i] = mean(ss[:, 3])
    original_wage_byage[i] = mean(original_data[
        (original_data[:, 2].==ages[i]).&(original_data[:, 3].==1),
        5,
    ])

end

plot(
    ages,
    simulated_wage_byage,
    label = "Simulated Data",
    lw = 3,
    ylim = (10000, 50000),
    main = "Wage",
)
plot!(ages, original_wage_byage, label = "Original Data", lw = 3)
title!("Wage")
################################################################################
#5. The 2x2 table of labor force participation by lagged labor force
#   participation.
################################################################################
ages = 45:54

function lfp_matrix(data, ages)
    matrix = zeros((2, 2))
    N = size(data, 1)
    lfp00 = 0
    lfp01 = 0
    lfp10 = 0
    lfp11 = 0
    for a = 1:length(ages)
        if ages[a] == 45.0
            lfp_t1 = data[data[:, 2].==45.0, 3]
            lfp_t0 = data[data[:, 2].==45.0, 7]
            cat = hcat(lfp_t0, lfp_t1)
            for k = 1:size(cat, 1)
                lfp00 += (1 - cat[k, 1]) * (1 - cat[k, 2])
                lfp01 += (1 - cat[k, 1]) * (cat[k, 2])
                lfp10 += (cat[k, 1]) * (1 - cat[k, 2])
                lfp11 += (cat[k, 1]) * (cat[k, 2])
            end
        else
            lfp_t1 = data[data[:, 2].==ages[a], 3]
            lfp_t0 = data[data[:, 2].==ages[a-1], 3]
            cat = hcat(lfp_t0, lfp_t1)
            for k = 1:size(cat, 1)
                lfp00 += (1 - cat[k, 1]) * (1 - cat[k, 2])
                lfp01 += (1 - cat[k, 1]) * (cat[k, 2])
                lfp10 += (cat[k, 1]) * (1 - cat[k, 2])
                lfp11 += (cat[k, 1]) * (cat[k, 2])
            end
        end
    end
    matrix[1, 1] = lfp00 / (lfp00 + lfp01)
    matrix[1, 2] = lfp01 / (lfp00 + lfp01)
    matrix[2, 1] = lfp10 / (lfp10 + lfp11)
    matrix[2, 2] = lfp11 / (lfp10 + lfp11)

    matrix
end

@show lfp_matrix_simul = lfp_matrix(simulated_data, ages);
@show lfp_matrix_orig = lfp_matrix(original_data, ages);

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
ages = 55:64

simulated_LFP_old = zeros(length(ages))
simulated_LFP_old1 = zeros(length(ages))
simulated_LFP_old2 = zeros(length(ages))
simulated_LFP_old3 = zeros(length(ages))
simulated_LFP_old4 = zeros(length(ages))

for i = 1:length(ages)
    ### Overall ###
    simulated_LFP_old[i] =
        mean((simulated_data[simulated_data[:, 2].==ages[i], 3] .== 1.0))

    ### By education ###

    simulated_LFP_old1[i] = mean((
        simulated_data[
            (simulated_data[:, 2].==ages[i]).&(simulated_data[:, 6].<12),
            3,
        ] .== 1.0
    ))
    simulated_LFP_old2[i] = mean((
        simulated_data[
            (simulated_data[:, 2].==ages[i]).&(simulated_data[:, 6].==12),
            3,
        ] .== 1.0
    ))
    simulated_LFP_old3[i] = mean((
        simulated_data[
            (simulated_data[
                :,
                2,
            ].==ages[i]).&(simulated_data[
                :,
                6,
            ].>12).&(simulated_data[:, 6].<16),
            3,
        ] .== 1.0
    ))
    simulated_LFP_old4[i] = mean((
        simulated_data[
            (simulated_data[:, 2].==ages[i]).&(simulated_data[:, 6].>15),
            3,
        ] .== 1.0
    ))

    simulated_LFP_old1[i] = mean((simulated_data[(simulated_data[:, 2].== ages[i]) .& (simulated_data[:,6] .< 12), 3] .== 1.0))
    simulated_LFP_old2[i] = mean((simulated_data[(simulated_data[:, 2].== ages[i]) .& (simulated_data[:,6] .== 12), 3] .== 1.0))
    simulated_LFP_old3[i] = mean((simulated_data[(simulated_data[:, 2].== ages[i]) .& (simulated_data[:,6] .> 12) .& (simulated_data[:,6] .< 16), 3] .== 1.0))
    simulated_LFP_old4[i] = mean((simulated_data[(simulated_data[:, 2].== ages[i]) .& (simulated_data[:,6] .> 15), 3] .== 1.0))


end

#Average years worked all women
sum(simulated_LFP_old)
#Average years worked less than high school
sum(simulated_LFP_old1)
#Average years worked high school
sum(simulated_LFP_old2)
#Average years worked some colle college
sum(simulated_LFP_old3)
#Average years worked college
sum(simulated_LFP_old4)



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

## Before tax values
simulated_beforetax_data = get_simulated_data(xhat, 20; τ1 = 0, τ2 = 0)

ages = 45:54

simulated_beforetax_LFP_old = zeros(length(ages))

for i = 1:length(ages)
    ### Overall ###
    simulated_beforetax_LFP_old[i] = mean((
        simulated_beforetax_data[simulated_beforetax_data[:, 2].==ages[i], 3] .== 1.0
    ))

end

@show sum(simulated_beforetax_LFP_old);

ages = 55:64

simulated_beforetax_LFP_old = zeros(length(ages))

for i = 1:length(ages)
    ### Overall ###
    simulated_beforetax_LFP_old[i] = mean((
        simulated_beforetax_data[simulated_beforetax_data[:, 2].==ages[i], 3] .== 1.0
    ))

end

@show sum(simulated_beforetax_LFP_old);

## After tax values
simulated_tax10_data = get_simulated_data(xhat, 20; τ1 = 0.1, τ2 = 0.1)

ages = 45:54

simulated_tax10_LFP_old = zeros(length(ages))

for i = 1:length(ages)
    ### Overall ###
    simulated_tax10_LFP_old[i] = mean((
        simulated_tax10_data[simulated_tax10_data[:, 2].==ages[i], 3] .== 1.0
    ))

end

@show sum(simulated_tax10_LFP_old);

ages = 55:64

simulated_tax10_LFP_old = zeros(length(ages))

for i = 1:length(ages)
    ### Overall ###
    simulated_tax10_LFP_old[i] = mean((
        simulated_tax10_data[simulated_tax10_data[:, 2].==ages[i], 3] .== 1.0
    ))

end

@show sum(simulated_tax10_LFP_old);


################################################################################
#3. Suppose the tax schedule is instead progressive.  The couple pays a 10
#   percent tax on the first 50,000 and a 20 percent tax on anything above that.
#   Assume that the couples reported the woman’s earnings accurately to the IRS,
#   although the wage is misreported in our data. You should therefore use your
#   estimate of the true wage for this exercise.a) What will happen to the
#   average number of years worked between the ages 45-54?   What is the total
#   revenue the IRS will collect? (b)  Do the same for ages 55-64.
################################################################################

## Before tax values
simulated_beforetax1020_data = get_simulated_data(xhat, 20;
                                            τ1 = 0, τ2 = 0, L = 50000)

ages = 45:54

simulated_beforetax1020_LFP_old = zeros(length(ages))

for i = 1:length(ages)
    ### Overall ###
    simulated_beforetax1020_LFP_old[i] = mean((
        simulated_beforetax1020_data[simulated_beforetax1020_data[:, 2].==ages[i], 3] .== 1.0
    ))

end


@show sum(simulated_beforetax1020_LFP_old);

ages = 55:64

simulated_beforetax1020_LFP_old = zeros(length(ages))

for i = 1:length(ages)
    ### Overall ###
    simulated_beforetax1020_LFP_old[i] = mean((
        simulated_beforetax1020_data[simulated_beforetax1020_data[:, 2].==ages[i], 3] .== 1.0
    ))

end

@show sum(simulated_beforetax1020_LFP_old);

## After tax values
simulated_tax1020_data = get_simulated_data(xhat, 20;
                                            τ1 = 0.1, τ2 = 0.2, L = 50000)

ages = 45:54

simulated_tax1020_LFP_old = zeros(length(ages))

for i = 1:length(ages)
    ### Overall ###
    simulated_tax1020_LFP_old[i] = mean((
        simulated_tax1020_data[simulated_tax1020_data[:, 2].==ages[i], 3] .== 1.0
    ))

end


@show sum(simulated_tax1020_LFP_old);

ages = 55:64

simulated_tax1020_LFP_old = zeros(length(ages))

for i = 1:length(ages)
    ### Overall ###
    simulated_tax1020_LFP_old[i] = mean((
        simulated_tax1020_data[simulated_tax1020_data[:, 2].==ages[i], 3] .== 1.0
    ))

end

@show sum(simulated_tax1020_LFP_old);
