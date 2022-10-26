
# Archive

## Motivation to archive
* `jeoung2021math`
  * Dataset has 57 features that makes it hard to run data synthesis and observe any results: 
  CTGAN took more than 24 hours to run while PrivBayes would take much more;
  * Dataset contains continuous features: it increases the execution time greatly. Moreover, some
  synthesizers don't support continuous variables and required discretization.
