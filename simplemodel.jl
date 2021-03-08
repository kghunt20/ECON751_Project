using Random, Statistics, Distributions, Optim

#generate_data1 and likehood1 are for the non-dynamic
#one period model

function generate_data1(σ_ϵ, N_s, σ_η, r_s, α1, α0)

    ϵ = σ_ϵ.*randn(1000)
    s = rand(1:N_s,1000)
    η = σ_η.*randn(1000)

    w = exp.(r_s.*s + ϵ)
    w_O = exp.(log.(w)+η)

    y = 5 .+ randn(1000)

    U1 = y .+ w
    U0 = y + α1.*y .+ α0

    P = U1 .> U0
    w_O = w_O .* P;
    
    w_O, P, s, y
    
end




function likelihood1(x; w_O = w_O, P = P, s = s, y = y)
    
    σ_ϵ = x[1]
    σ_η = x[2]
    r = x[3]
    α1 = x[4]
    α0 = x[5]
    
    u = log.(w_O + (1.0.-P)) - r.*s
    σ_u = (σ_ϵ^2 + σ_η^2)^0.5
    ρ = σ_ϵ/σ_u
    
    ϵ_star = log.(α1.*y .+ α0) .- r.*s
    A = (ϵ_star .- ρ*σ_ϵ/σ_u.*u)./(σ_ϵ*(1-ρ^2)^0.5)
    B = pdf.(Normal(), u./σ_u)./σ_u
    
    probabilities = zeros(1000, 2)
    
    probabilities[:,1] = cdf.(Normal(), ϵ_star./σ_ϵ)
    probabilities[:,2] = (1.0.-cdf.(Normal(), A)).*B
    
    -1.0 .* sum((1.0.-P).*log.(probabilities[:,1]) + P.*log.(probabilities[:,2]))
    
end

x1 = [1.0, 1.0, 0.01, 0.1, 0.5]

w_O, P, s, y = generate_data1(1.0, 8, 1.0, 0.01, 0.1, 0.5)

optimize(likelihood1, x1)


##########################################################################################
#Here I create some code for the dynamic problem with simple utility and wage
#structure
##########################################################################################


#Model Functions

#σ_ϵ = x[1]
#σ_η = x[2]
#r_s = x[3]
#r_k = x[4]
#α1 = x[5]
#α0 = x[6]
#s = schooling
#k = experience

function log_wage_det(x, s, k)
    
    r_s = x[3]
    r_k = x[4]
    
    r_s*s + r_k*k
    
end

function c_U1(x, c_y, w, s, k)
    
    c_y + w
    
end

function c_U0(x, c_y, w, s, k)
    
    α1 = x[5]
    α0 = x[6]
    
    c_y + α1*c_y + α0
end
    
function generate_data2(x, N_obs; y = y, s = s, T = T, β = β, Δϵ=Δϵ, k_start = k_start)
   
    σ_ϵ = x[1]
    σ_η = x[2]
    r_s = x[3]
    r_k = x[4]
    α1 = x[5]
    α0 = x[6]
    
    ϵ_grid, P, ϵ_star = solve_model2(x; k_start = k_start)
    k_grid = k_start:(k_start+T-1)
    
    data_wO = zeros(N_obs, T)
    data_P = zeros(N_obs, T)
    
    for i_obs = 1:N_obs

        i_k = 1
    
        for t = 1:T

            ϵ = round(randn()*σ_ϵ, digits = digits = -Int(log10(Δϵ)))
            i_ϵ = findmin(abs.(ϵ_grid.-ϵ))[2]

            if P[i_ϵ, t, i_k] == 1

                data_P[i_obs, t] = 1
                data_wO[i_obs,t] = exp(log_wage_det(x, s, k_grid[i_k]) + ϵ_grid[i_ϵ] + σ_η*randn())
                i_k = i_k + 1
            else 
                data_P[i_obs,t] = 0
                data_wO[i_obs,t] = -9.0
            end

        end
    end
    
    data_wO, data_P
    
end   

function likelihood2(x; data_P = data_P, data_wO = data_wO, data_s = data_s, data_k_start = data_k_start, data_y = data_y, 
                        T = T, N_obs = N_obs, β = β)
    
    σ_ϵ = x[1]
    σ_η = x[2]
    r_s = x[3]
    r_k = x[4]
    α1 = x[5]
    α0 = x[6]
    
    LL = 0.0
    
    σ_u = (σ_ϵ^2 + σ_η^2)^0.5
    ρ = σ_ϵ/σ_u
    
    for i = 1:N_obs
        
        
        
        y = data_y[i,:]
        s = data_s[i]
        k_start = Int(data_k_start[i])
        k_grid = k_start:(k_start + T - 1)
        
        ϵ_grid, P, ϵ_star = solve_model2(x; k_start = k_start)
        
        i_k = 1
        
        for t = 1:T    
        
            if data_P[i,t] == 1

                u = log(data_wO[i,t]) - log_wage_det(x, s, k_grid[i_k])
                A = (ϵ_star[t, i_k] - ρ*σ_ϵ/σ_u*u)/(σ_ϵ*(1-ρ^2)^0.5)
                B = pdf(Normal(), u/σ_u)/σ_u
                
                LL -= log( (1.0-cdf(Normal(), A))*B )
                
                i_k += 1
                
            else
                
                LL -= log(cdf(Normal(), ϵ_star[t, i_k]/σ_ϵ))
                
            end
            
        end
        
    end
    
    @show LL
    
    LL
            
end    


x0 = [1.0, 1.0, 0.05, 0.1, 0.1, 0.5]
T = 20
β = 0.95
N_obs = 1000
Δϵ = 0.1

data_wO = zeros(N_obs, T)
data_P = zeros(N_obs, T)
data_y = zeros(N_obs, T)
data_s = zeros(N_obs)
data_k_start = zeros(N_obs)

for p = 1:N_obs
    
    y = 10.0 .*randn(20).+100
    data_y[p,:] = y
    k_start = rand(0:10)
    data_k_start[p] = k_start
    s = rand(8:16)
    data_s[p] = s
    data_wO[p,:], data_P[p,:] = generate_data2(x0, 1; k_start = k_start);
    
end
    

#Even just 100 iterations is so slow and not so close to true values!

optimize(likelihood2, x0, iterations = 100).minimizer
