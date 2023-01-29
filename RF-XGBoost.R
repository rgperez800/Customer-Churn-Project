## Decision Tree, Random Forest, XGBoost
library(rpart.plot)
library(vip)
set.seed(1115)

### Fit a classification tree

# Decision tree
tree_spec <- decision_tree() %>%
  set_engine("rpart")

# Classification decision
class_tree_spec <- tree_spec %>%
  set_mode("classification")

# Fit the model
class_tree_fit <- class_tree_spec %>%
  fit(attrition_flag ~ customer_age +
        gender +
        income_category +
        total_relationship_count +
        months_inactive_12_mon +
        credit_limit +
        total_revolving_bal +
        avg_open_to_buy +
        total_amt_chng_q4_q1 +
        total_trans_amt + 
        total_trans_ct +
        total_ct_chng_q4_q1 + 
        avg_utilization_ratio +
        card_category,
      data = churn_train)

# Visualize decision tree
class_tree_fit %>%
  extract_fit_engine() %>%
  rpart.plot(roundint = FALSE)
  

# Check training set accuracy
augment(class_tree_fit, new_data = churn_train) %>%
  roc_auc(truth = attrition_flag, estimate = .pred_0)

# Confusion matrix
augment(class_tree_fit, new_data = churn_train) %>%
  conf_mat(truth = attrition_flag, estimate = .pred_class)

# Check Testing Set Confusion Matrix
augment(class_tree_fit, new_data = churn_test) %>%
  conf_mat(truth = attrition_flag, estimate = .pred_class)

# ROC of testing set
augment(class_tree_fit, new_data = churn_train) %>%
  roc_auc(truth = attrition_flag, estimate = .pred_0)


# Estimate is good, but we will perform hyperparameter tuning in our decision tree to see if we can get a higher estimate
# Tune cost_complexity for a more optimal complexity
class_tree_wf <- workflow() %>%
  add_model(class_tree_spec %>% set_args(cost_complexity = tune())) %>%
  add_recipe(recipe_churn2)

# Use k-fold validation and develop grid of values
set.seed(1115)
param_grid <- grid_regular(cost_complexity(range = c(-5,-1)), levels = 10)

tune_res <- tune_grid(
  class_tree_wf,
  resamples = churn_folds,
  grid = param_grid,
  metrics = metric_set(roc_auc)
)

# Which values of cost_complexity produces the highest ROC_AUC?
autoplot(tune_res)

# Select best value of cost_complexity
best_complexity <- select_best(tune_res)
class_tree_final <- finalize_workflow(class_tree_wf, best_complexity)
class_tree_final_fit <- fit(class_tree_final, data = churn_train)

# Visualize model
class_tree_final_fit %>%
  extract_fit_engine() %>%
  rpart.plot(roundint = FALSE)

# Check performance on testing data
augment(class_tree_final_fit, new_data = churn_test) %>%
  roc_auc(truth = attrition_flag, estimate = .pred_0)  # Higher estimate


### Random Forest

## We will take a look at a random forest model
## Here, we will tune mtry, and min_n
## Using recipe 2, the one without PCA
## Random forest does not require PCA here, since it can see the relations, plus we can create a variable importance chart

# RF specification
# Warning: this gonna take a while!
set.seed(1115)
rf_spec <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_spec_wf <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(recipe_churn2)

reg_grid <- grid_regular(mtry(range = c(1,14)), min_n(range = c(1,35)), levels = 10)

# Tuned model
tune_forest <- tune_grid(
  rf_spec_wf,
  resamples = churn_folds,
  grid = reg_grid,
  metrics = metric_set(roc_auc)
  )

# Plot tuned model
autoplot(tune_forest)

# Choose best model and fit the data
best_forest <- select_best(tune_forest)
class_forest_final <- finalize_workflow(rf_spec_wf, best_forest)
class_forest_final_fit <- fit(class_forest_final, data = churn_train)

# Try testing set
augment(class_forest_final_fit, new_data = churn_test) %>%
  roc_auc(truth = attrition_flag, estimate = .pred_0)  

# RF Variable Importance Chart
vip(extract_fit_parsnip(class_forest_final_fit))


### Boosted Tree Model

set.seed(1115)
## We'll now fit a boosted tree model
## Set engine to XGBoost
## We'll be tuning, mtry, learn_rate, and min_n

# Boost specification
boost_spec <- boost_tree(mtry = tune(), min_n = tune(), learn_rate = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

boost_grid <- grid_regular(mtry(range = c(1,14)), min_n(range = c(1,15)), learn_rate(range = c(0.05,0.5)), levels = 10)

boost_spec_wf <- workflow() %>%
  add_model(boost_spec) %>%
  add_recipe(recipe_churn2)

# Tuned Boost
tune_boost <- tune_grid(
  boost_spec_wf,
  resamples = churn_folds,
  grid = boost_grid,
  metrics = metric_set(roc_auc)
)

# Plot tuned model
autoplot(tune_boost)

# Select best boosted model
best_boost <- select_best(tune_boost)
class_boost_final <- finalize_workflow(boost_spec_wf, best_boost)
class_boost_final_fit <- fit(class_boost_final, data = churn_train)

# Fit to testing set
augment(class_boost_final_fit, new_data = churn_test) %>%
  roc_auc(truth = attrition_flag, estimate = .pred_0)  

# Boosted Tree Variable Importance Chart
vip(extract_fit_parsnip(class_boost_final_fit))

