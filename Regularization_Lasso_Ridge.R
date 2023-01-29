### Ridge Regression
## This is where we will tune the penalty hyperparameter of our logistic regression model using the glmnet engine
## We will use glmnet engine to make this possible
## Will use recipe without PCA, since regularization already tries to reduce the number of features

library(glmnet)


set.seed(1115)
### Ridge Regression 

# Specify a logistic regression model
ridge_spec <- logistic_reg(mixture = 0, penalty = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# Create workflow object
ridge_wf <- workflow() %>%
  add_recipe(recipe_churn2) %>%
  add_model(ridge_spec)

# Create grid for values of penalty
penalty_grid <- grid_regular(penalty(range = c(-30,20)), levels = 10)
penalty_grid

tune_ridge <- tune_grid(
  ridge_wf,
  resamples = churn_folds,
  grid = penalty_grid
)

# Plot tuned model
autoplot(tune_ridge)

# See raw metrics
collect_metrics(tune_ridge)

# Select best
best_penalty <- select_best(tune_ridge, metric = "roc_auc")
best_penalty

# Finalize workflow
ridge_final <- finalize_workflow(ridge_wf, best_penalty)
ridge_final_fit <- fit(ridge_final, data = churn_train)

ridge_final_fit %>%
  extract_fit_engine() %>%
  plot(xvar = "lambda")


# Apply model to testing set
augment(ridge_final_fit, new_data = churn_test) %>%
  roc_auc(truth = attrition_flag, estimate = .pred_0) 

