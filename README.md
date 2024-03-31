# learn_opt
 learn optimisation

- [Bayesian_opt](https://github.com/xueannafang/learn_opt/blob/main/Bayesian_opt.ipynb)
A step-by-step learning diary for Bayesian basics.

-[More about acq functions](https://github.com/xueannafang/learn_opt/blob/main/more_about_acq_func_and%20some_stats.ipynb).
A few introduction about working mechanism of different acquisition functions.

- [Bayes_opt_class](https://github.com/xueannafang/learn_opt/blob/main/Bayes_opt_class.ipynb)
The envelopped workflow. Allow optimisation for complicated imaginary function. Allow automatic selection of next data.

- [Deal with real-life experimental data - gp-ucb - without theoretical model](https://github.com/xueannafang/learn_opt/blob/main/real_life_test_multi_dim_PCA_GP_UCB.ipynb)
No theoretical model was hypothesised now. Try to compare the Bayesian prediction from training dataset with observed best point. This version also includes dimensionality reduction (from 2 to 1, according to weighted contribution). A different kernel (Matern) was attempted. More functions on hyperparameter modification..
(And we can compare with other standard BO package using the same data - [working example](https://github.com/xueannafang/learn_opt/blob/main/multi_dim_BO_by_others.ipynb).)

A version without weighting variables is [here](https://github.com/xueannafang/learn_opt/blob/main/real_life_test_one_dim_PCA_GP_UCB.ipynb). 


Key steps in [Bayes_opt_class](https://github.com/xueannafang/learn_opt/blob/main/Bayes_opt_class.ipynb):

1) Specify the hyperparameters in the user input block:

```
#user inputs and pre-define hyperparameters

noise_level = 0.1 #the expected standard deviation of experimental measurement

#hyperparameters for rbf
rbf_hp = {
    'length_scale': 1,
    'length_scale_bounds' : (1e-2, 1e2),
    'amp' : 1,
    'const' : 0
}

kernel = rbf_hp['amp']*RBF(length_scale = rbf_hp['length_scale'], length_scale_bounds = rbf_hp['length_scale_bounds']) + rbf_hp['const']

#hyperparameters and kernel dictionary
gpr_hp_dict = {
    'alpha' : noise_level,
    'kernel' : kernel,
    'random_state' : 1
}

#X's to predict by GPR
#see the X = linspace part in the next cell 
```

2) Prepare the initial experimental data ```X_real```, ```Y_real``` (in the format of np.ndarray).

(The example data was created by the imagnary function block)

```
#generate initial (imaginary) data

def std_model(X:np.ndarray)->np.ndarray:
    """
    This is the standard model without noise
    """
    Y = 0.1*X**2 + np.sin(X) + 0.2*X + 10 #This is a pre-defined real physical model for us to compare with the prediction results
#     Y = 0.001*X**4-0.1*X**2 + 0.2*X + 10 # an arbitrary function, as long as it has a minimal
    return Y

def exp_model(X:np.ndarray, noise_level = 0.1):
    #noise_level is the standard deviation, 67% of the data will be within pm 1 from the centre y data
    Y = std_model(X)
    noise = np.random.normal(scale = noise_level, size = Y.shape)
    real_Y = Y + noise
    return real_Y

#generate the initial X
X = np.linspace(-10, 10, 201) #Generate 201 initial points from -10 to 10
#This is X_to_predict in the class later
Y = std_model(X)
real_X = np.random.uniform(-10, 10, 11) # this is the perturbated series.
real_Y = exp_model(real_X, noise_level = noise_level)


#plot initial dataset
fig, ax = plt.subplots()
ax.plot(X, Y, color = 'red', label = 'theoretical')
ax.scatter(real_X, real_Y, color = 'blue', marker = 'x', label = 'experimental')
ax.legend()


#print measurements data
print(f"measured X: {real_X}")
print(f"measured Y: {real_Y}")
```


3) Create an instance of BayesOpt class and load data, run the first cycle of optimisation.

(Check the ```testblock``` cell)

```
#testblock
#print("real_X, real_Y")
#print(real_X, real_Y)

x0 = 0.5

bo = BayesOpt()
#print("load_data, after dimension conversion:")

#load initial data
bo.load_data(real_X, real_Y)

#first round of gpr
bo.do_gpr(gpr_hp_dict)
bo.gpr_predict(X)
#check the GPR results make sense or not. If yes, please continue with aqusition functions
#Otherwise, modify the gpr_hp_dict inputs

#do prediction for initial guess x0
bo.aq_func(x0, ucb_beta = 2)

```

4) To add new data into the prior knowledgebase, there are two ways: manual or auto.

For the maunal version, do:

```
new_x = np.array([0.004]) #here 0.004 is an example new_x value
new_y = exp_model(new_x, noise_level = noise_level) # in real case, the new_y would also be an array specified by user, but here its based on imagniary model, so its auto generated.
bo.add_new_data(new_x, new_y)
```

then run the optimisation part again, see what has been changed:

```
bo.do_gpr(gpr_hp_dict)
bo.gpr_predict(X)
bo.aq_func(x0, ucb_beta = 1.96)
```


You can check the new data has been added to the dataset or not by the length of data_X:

```
len(bo.data_X)
```

The auto version will be introduced in the ```auto_forward_until_convg()``` part.

5) To make prediction based on a given initial guess x0, use the ```forward()``` function.

```
forward(x0)
```

which expected to return a plot and the next optimised x to go.

6) For automatic prediction (and selection of x0 based on the previous suggested "next x to go"), use:

```
auto_forward_until_convg(initial_x0)
```

If the prediction is successful, a plot with final status will be presented, with all optimisation steps saved in ```gpr_log``` dictionary:


```
gpr log: {'gpr': GaussianProcessRegressor(alpha=0.1, kernel=1**2 * RBF(length_scale=1) + 0**2,
                         random_state=1), 'opt_kernel': 15.4**2 * RBF(length_scale=10.1) + 0.00319**2, 'score': 0.9824295997805579}
```

The kernel can be extracted by:

```
bo.gpr_log['opt_kernel']
```

Here is an example workflow, see how the kernel and the plot changed in each step:

<p>
 <img src=https://github.com/xueannafang/learn_opt/blob/main/exp_1.png width=1000>
 </p>

For the automatic workflow, user only needs to specify the first guess of x0, then the rest of "x0" will be determined by the "suggested next x" from the previous step.

The optimisation will continue until two "suggested next x" reached the same, which means the model has converged.
 
<p>
 <img src=https://github.com/xueannafang/learn_opt/blob/main/exp_2.png width=1000>
 </p>




