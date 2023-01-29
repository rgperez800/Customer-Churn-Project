### Support Vector Machine
## We will use a linear specification

library(kernlab)
set.seed(1115)

# Specification
svm_lin_spec <- svm_linear(cost = tune(), margin = 0.1) %>%
  set_mode("classification") %>%
  set_engine("kernlab", scaled = FALSE)

# Workflow
svm_lin_wf <- workflow() %>%
  add_model(svm_lin_spec) %>%
  add_recipe(recipe_churn)

# Grid
cost_grid <- grid_regular(cost(), levels = 10)

# Tuning model
tune_svm_lin <- tune_grid(
  svm_lin_wf,
  resamples = churn_folds,
  grid = cost_grid
)

# Autoplot
autoplot(tune_svm_lin)


# Finalize workflow
best_cost <- select_best(tune_svm_lin, metric = "roc_auc")
svm_linear_final <- finalize_workflow(svm_lin_wf, best_cost)
svm_linear_final_fit <- fit(svm_linear_final, data = churn_train)




# Apply to testing data
augment(svm_linear_final_fit, new_data = churn_test) %>%
  roc_auc(truth = attrition_flag, estimate = .pred_0)





