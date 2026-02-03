import rpy2.robjects.packages as rpackages
from rpy2.robjects import r

utils = rpackages.importr('utils')
# utils.install_packages("reticulate")
# if not rpackages.isinstalled("remotes"):
#     utils.install_packages("remotes")
# 
# r('remotes::install_github("quanteda/spacyr@HEAD")')
# utils.install_packages('tm', dependencies=True)
# utils.install_packages('quanteda', dependencies=True)
utils.install_packages("quanteda.textstats")
# utils.install_packages('politeness', dependencies=True)