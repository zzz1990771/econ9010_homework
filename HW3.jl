# generate fake data and plot
Pkg.add("GR")
using Distributions, Plots, PlotRecipes, StatPlots
gr()


a = 10
b = 10

#sigma=0.5
srand(13457)
n = 100
sigma2 = 2
sigma = sqrt(sigma2)
x = rand(Normal(1, 2), n)
u = rand(Normal(0, sigma), n)
y = a + b*x + u
scatter([x y],palette = :blues, alpha = 0.3,legend=false,gridcolor=:grey,
             xlab = "x", ylab = "y")
std(u)
mean(y)

# load the functions
include("bayesreg.jl")
# use the function to get output:
bhat, seb, s2hat, Vb, R2, yhat = bayesreg(y,x)

bhat, seb, s2hat, Vb, R2, yhat = bayesregNIG(y,x)

# informative prior
b0 = [10. 10.]'
B0 = [1. 0; 0 1.]
a0 = 3.
d0 = 3.
b1, seb1, s21, Vb1, R21, yhat1 = bayesregNIG(y,x,b0=b0,B0=B0,a0=a0,d0=d0)

# draw from the marginal posterior for beta
bdraws = marg_post_mu(b1[2], seb1[2],(n-2))
# plot the posterior
plot(bdraws, st=:density,color=:blue,fill=(0,:blue),alpha=0.3,label="posterior for beta",size=(800,400))
plot!(bdraws, st=:histogram,normed=true, bins=100, color=:orange,alpha=0.4,label="")

#bs = 8.0:0.01:12.0  # evaluate pdf of centralized t
t = rand(TDist(n-2),10000000)

beta = seb[2]*t + b1[2]  # scale for posterior of beta

plot!(bs,beta,st=:density,linecolor=:green,linewidth=2,label="analytical")




#gibbs sampling
include(".\\gsreg\\gsreg.jl")
X = [ones(n) x]
# uninformative prior
bs, s2s = gsreg(y,X)
mean(X[:,2])

mean(bs[:,1])
std(bs[:,1])
mean(bs[:,2])
std(bs[:,2])
mean(s2s)
std(s2s)

include(".\\gsreg\\mcmc_sample_plot.jl")
mcmc_sample_plot(bs[:,2])
vline!([0.0 10.0], color = :green, label=["" "true mean"])

mcmc_sample_plot(bs[:,1])

mcmc_sample_plot(s2s)
vline!([0.0 sigma^2], color = [:black :green], label=["" "true mean"])

# Informative prior
b0 = [10.0; 10.0]  # b0 must be a column vector!
iB0 = [1.0 0.0; 0.0 1.0]
d0 = 0.1
a0 = 0.01

# tau is starting value
bs, s2s = gsreg(y,X, M = 10000,b0=b0,iB0=iB0,d0=d0,a0=a0,tau=0.1)

mcmc_sample_plot(bs[:,2])
vline!([0.0 10.0], color = :green, label=["" "true mean"])

mcmc_sample_plot(s2s)
vline!([0.0 sigma^2], color = [:black :green], label=["" "true mean"])
