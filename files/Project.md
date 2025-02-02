# ASP Course Project
* Course project is due on 11.15 (Friday) night.

## [Repository](https://github.com/pyfe/pyfeng) and [Project List](https://github.com/pyfe/pyfeng/projects)

## Presentation
### 5.07 Fri
*

## Do do
* Create new project wiht **title**, **short description**, and **team**.

## Topics
* Among the topics and HWs covered in the class, choose an in-depth research on one topic. You are also welcome to do the project on your own original idea. Otherwise, pick one from my suggestions which are basically understanding and implementing literatures. Topics includes 
  * Spread/Basket/Asian option pricing
  * SABR other stochastic volatility models
  * Copula

## File requirements
* Core implementation (.py): python class and functions
  * Make sure to add docstring in detail. Example: [bsm.py](https://github.com/PHBS/pyfeng/blob/master/pyfeng/bsm.py)
  * __Integrate into the `pyfeng` folder.__ Make pull-requests (pr).
* Documentation/Test (__.ipynb__): one Jupyter notebook file briefly describing the method (base theory, equations, SDE, strength/weakness, etc), the function prototype and arguments (manual style) and the usage examples
  * The best examples are from numpy documentation: [example](https://docs.scipy.org/doc/numpy-1.10.1/reference/routines.polynomials.hermite.html)
  * Also include examples and tests of your class (e.g., test against reference parameters)
  * __Put into the `test` folder. The file name should be Test_YOURPROJECT.ipynb__ 

## Other guidelines
* Deadline: 5.9 (Sunday)

## Suggested Topics and Papers 
* [Basket/Asian option] Ju's Taylor expansion method
  * Ju, N. (2002). Pricing Asian and Basket Options Via Taylor Expansion. Journal of Computational Finance, 5(3), 79–103. https://doi.org/10.21314/JCF.2002.088
  * (Performance Comparison) Krekel, M., de Kock, J., Korn, R., & Man, T.-K. (2004). An analysis of pricing methods for basket options. Wilmott Magazine, 2004(7), 82–89. See the [notebook](https://github.com/PyFE/PyfengForPapers/blob/main/ipynb/KrekelEtAl2004-Wilmott-BasketOption.ipynb) in PyfengForPapers.
  * Implement the method in a new class. In python notebook, summarize the method, write a quick help and report strength and weakness.
* [Basket/Asian option] Johnson's SU distribution
  * Posner, S. E., & Milevsky, M. A. (1998). Valuing exotic options by approximating the SPD with higher moments. The Journal of Financial Engineering, 7(2).
  * Also see Ju (2002) for perfomance comparison
* [Asian option] Tight lower bound
  * Rogers, L. C. G., & Shi, Z. (1995). The value of an Asian option. Journal of Applied Probability, 32(4), 1077–1088.
* [Exact Asian Option Pricing] 
  * Linetsky, V. (2004). Spectral Expansions for Asian (Average Price) Options. Operations Research, 52(6), 856–867. https://doi.org/10.1287/opre.1040.0113
* Conditional Monte-Carlo for SV models
  * OUSV, Garch Diffusion (and 3/2)
  * Ornstein Uhlenbeck Simulation. ([Vasicek Model](https://en.wikipedia.org/wiki/Vasicek_model) for interest rate)
* Almost-Exact Simulation for
  * Heston
  * 3/2
  * Garch Diffusion


### * Simulations of Heston model
* [Heston Model](https://en.wikipedia.org/wiki/Heston_model)
* ~Broadie, M., Kaya, Ö., 2006.~ Exact Simulation of Stochastic Volatility and Other Affine Jump Diffusion Processes. Operations Research 54, 217–231. https://doi.org/10.1287/opre.1050.0247 | Glasserman, P., Kim, K.-K., 2011. Gamma expansion of the Heston stochastic volatility model. Finance Stoch 15, 267–296. https://doi.org/10.1007/s00780-009-0115-y
* ~Andersen, L., 2008.~ Simple and efficient simulation of the Heston stochastic volatility model. The Journal of Computational Finance 11, 1–42. https://doi.org/10.21314/JCF.2008.189
* Kahl, C., Jäckel, P., 2006. Fast strong approximation Monte Carlo schemes for stochastic volatility models. Quantitative Finance 6, 513–536. https://doi.org/10.1080/14697680600841108
* Lord, R., Koekkoek, R., Dijk, D.V., 2010. A comparison of biased simulation schemes for stochastic volatility models. Quantitative Finance 10, 177–194. https://doi.org/10.1080/14697680802392496

### * Rough Volatility
* McCrickerd, R., Pakkanen, M.S., 2018. Turbocharging Monte Carlo pricing for the rough Bergomi model. Quantitative Finance 18, 1877–1886. https://doi.org/10.1080/14697688.2018.1459812
* Bennedsen, M., Lunde, A., Pakkanen, M.S., 2017. Hybrid scheme for Brownian semistationary processes. Finance Stoch 21, 931–965. https://doi.org/10.1007/s00780-017-0335-5
* J. Gatheral's [python code](https://tpq.io/p/rough_volatility_with_python.html)

### * GARCH-diffusion model
* Barone-Adesi, G., Rasmussen, H., Ravanelli, C., 2005. An option pricing formula for the GARCH diffusion model. Computational Statistics & Data Analysis, 2nd CSDA Special Issue on Computational Econometrics 49, 287–310. https://doi.org/10.1016/j.csda.2004.05.014
* PhD Thesis of Ravanelli, C., University of Lugano, Switzerland, https://doc.rero.ch/record/4229/files/1_2003ECO001.pdf
* Papadopoulos, Y.A., Lewis, A.L., 2018. A First Option Calibration of the GARCH Diffusion Model by a PDE Method. arXiv:1801.06141 [q-fin].

### * Exact Simulation Scheme
* __General SDE__: Beskos, A., Roberts, G.O., 2005. Exact simulation of diffusions. Ann. Appl. Probab. 15, 2422–2444. https://doi.org/10.1214/105051605000000485
* __OU SV Model__: Li, C., Wu, L., 2019. Exact simulation of the Ornstein–Uhlenbeck driven stochastic volatility model. European Journal of Operational Research 275, 768–779. https://doi.org/10.1016/j.ejor.2018.11.057
* __Heston Model__: Broadie, M., Kaya, Ö., 2006. Exact Simulation of Stochastic Volatility and Other Affine Jump Diffusion Processes. Operations Research 54, 217–231. https://doi.org/10.1287/opre.1050.0247 | Glasserman, P., Kim, K.-K., 2011. Gamma expansion of the Heston stochastic volatility model. Finance Stoch 15, 267–296. https://doi.org/10.1007/s00780-009-0115-y
* __3/2 SV Model__: Baldeaux, J., 2012. Exact simulation of the 3/2 model. Int. J. Theor. Appl. Finan. 15, 1250032. https://doi.org/10.1142/S021902491250032X
* __SABR Model__: Cai, N., Song, Y., Chen, N., 2017. Exact Simulation of the SABR Model. Operations Research 65, 931–951. https://doi.org/10.1287/opre.2017.1617
* __Computing Moments from Laplace Transform__: Choudhury, G.L., Lucantoni, D.M., 1996. Numerical Computation of the Moments of a Probability Distribution from its Transform. Operations Research 44, 368–381. https://doi.org/10.1287/opre.44.2.368

### * CEV Model
* Option pricing under Constant Elasticity of Variance (CEV) model (formula available many on-line sources):
implement the method, include it to Normal class and write a thorough test code. In python notebook, summarize the method, write a quick help and report strength and weakness.
* Basic model is completed. Need to check Greeks (Delta/Gamma/Vega/Theta)
