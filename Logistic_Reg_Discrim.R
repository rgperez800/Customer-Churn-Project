# Logistic Regression, Linear Discriminant Analysis, Quadratic Discriminant Analysis, Naive Bayes Models
# Overall, logistic model performed the best

# Use first recipe, with PCA
set.seed(1115)

recipe_churn <- recipe(attrition_flag ~ customer_age + gender + income_category + total_relationship_count + months_inactive_12_mon + credit_limit + total_revolving_bal + avg_open_to_buy + total_amt_chng_q4_q1 + total_trans_amt + total_trans_ct + total_ct_chng_q4_q1 + card_category + avg_utilization_ratio, data = churn_train) %>%
  # We only include 14 predictors
  # Credit limit needed to be logged, as it had a right-skewed distribution
  # We needed to categorize some of our categorical variables, use one-hot encoding
  # Must scale and center all predictors, as our predictors measure different values
  # Impute income_category via k-nearest neighbors to remove all nulls
  # Up-sample using SMOTE to deal with imbalanced data
  # We will be applying PCA to our models to reduce dimensionality 
  step_impute_knn(income_category, impute_with = imp_vars(all_nominal_predictors())) %>%
  step_log(credit_limit) %>%
  step_dummy(gender, one_hot = TRUE) %>%
  step_dummy(card_category, one_hot = TRUE) %>%
  step_dummy(income_category, one_hot = TRUE) %>%
  step_smote(attrition_flag, over_ratio = 0.4765) %>%
  step_scale(all_predictors()) %>%
  step_center(all_predictors()) %>%
  step_pca(all_predictors()) 



### K-Fold Cross Validation
set.seed(1115)
churn_folds <- vfold_cv(churn_train, v = 10, strata = attrition_flag)  # 10-fold CV



## Model Building


### Logistic Regression Model
set.seed(1115)
# Specify a logistic regression model
log_reg <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# Create a workflow
log_wkflow <- workflow() %>%
  add_model(log_reg) %>%
  add_recipe(recipe_churn)

# Fit the model by applying the workflow to the training set
log_fit <- fit(log_wkflow, churn_train)


# Confusion matrix
set.seed(1115)
augment(log_fit, new_data = churn_train) %>%
  conf_mat(truth = attrition_flag, estimate = .pred_class)




# ROC_AUC of model
set.seed(1115)
log_reg_acc <- augment(log_fit, new_data = churn_train) %>%
  roc_auc(attrition_flag, estimate = .pred_0)
log_reg_acc



### Linear Discriminant Analysis

set.seed(1115)

lda_mod <- discrim_linear() %>%
  set_mode("classification") %>%
  set_engine("MASS") 

lda_wkflow <- workflow() %>%
  add_model(lda_mod) %>%
  add_recipe(recipe_churn)

lda_fit <- fit(lda_wkflow, churn_train)

predict(lda_fit, new_data = churn_train, type = "prob")




# Confusion matrix
set.seed(1115)
augment(lda_fit, new_data = churn_train) %>%
  conf_mat(truth = attrition_flag, estimate = .pred_class)




set.seed(1115)
# ROC AUC

lda_acc <- augment(lda_fit, new_data = churn_train) %>%
  roc_auc(attrition_flag, estimate = .pred_0)
lda_acc

### Quadratic Discriminant Analysis
set.seed(1115)
qda_mod <- discrim_quad() %>%
  set_mode("classification") %>%
  set_engine("MASS")

qda_wkflow <- workflow() %>%
  add_model(qda_mod) %>%
  add_recipe(recipe_churn)

qda_fit <- fit(qda_wkflow, churn_train)



# QDA performance
predict(qda_fit, new_data = churn_train, type = "prob")


# Confusion matrix
augment(qda_fit, new_data = churn_train) %>%
  conf_mat(truth = attrition_flag, estimate = .pred_class)



qda_acc <- augment(qda_fit, new_data = churn_train) %>%
  roc_auc(attrition_flag, estimate = .pred_0)
qda_acc




set.seed(1115)
### Naive Bayes
# Our worst model, it will be excluded in our final model building
nb_mod <- naive_Bayes() %>%
  set_mode("classification") %>%
  set_engine("klaR") %>%
  set_args(usekernel = FALSE)

nb_wkflow <- workflow() %>%
  add_model(nb_mod) %>%
  add_recipe(recipe_churn)

nb_fit <- fit(nb_wkflow, churn_train)



set.seed(1115)
predict(nb_fit, new_data = churn_train, type = "prob")



augment(nb_fit, new_data = churn_train) %>%
  conf_mat(truth = attrition_flag, estimate = .pred_class)



set.seed(1115)
nb_acc <- augment(nb_fit, new_data = churn_train) %>%
  roc_auc(attrition_flag, estimate = .pred_0)
nb_acc



## Compare model performance


rocs <- c(log_reg_acc$.estimate, lda_acc$.estimate, nb_acc$.estimate, qda_acc$.estimate)

models <- c("Logistic Regression", "LDA", "Naive Bayes", "QDA")
results <- tibble(rocs = rocs, models = models)
results %>%
  arrange(-rocs)

## Quad analysis is our best model!

# Fitting to Testing Data

predict(qda_fit, new_data = churn_test, type = "prob")

# Confusion matrix

augment(qda_fit, new_data = churn_test) %>%
  conf_mat(truth = attrition_flag, estimate = .pred_class) %>%
  autoplot(type = "heatmap")

# Look at testing accuracy
multi_metric <- metric_set(roc_auc)

augment(qda_fit, new_data = churn_test) %>%
  multi_metric(truth = attrition_flag, .pred_0)

## ROC Curve
augment(qda_fit, new_data = churn_test) %>%
  roc_curve(attrition_flag, .pred_0) %>%
  autoplot()

### Apply resamples to each model: logistic, linear, quad
log_fit_resample <- fit_resamples(log_wkflow, churn_folds)

lda_fit_resample <- fit_resamples(resamples = churn_folds,
                                  lda_wkflow)

qda_fit_resample <- fit_resamples(qda_wkflow, resamples = churn_folds)

# Which model performed the best?
collect_metrics(log_fit_resample)
collect_metrics(lda_fit_resample)
collect_metrics(qda_fit_resample)

# Logistic and LDA have similar stats; choose logistic because data is better suited for it

# Choose logistic model
log_test <- fit(log_wkflow, churn_test)
predict(log_test, new_data = churn_test, type = "prob") %>% 
  bind_cols(churn_test %>% dplyr::select(attrition_flag)) %>% 
  roc_auc(truth = attrition_flag, estimate = .pred_0)

