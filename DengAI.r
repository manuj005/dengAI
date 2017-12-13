# Reading the datasets
train_data <- read.csv('dengue_features_train.csv')
train_label <- read.csv('dengue_labels_train.csv')
test_data <- read.csv('dengue_features_test.csv')

# Combining the featureset and labelset
train_total <- cbind(train_data, train_label['total_cases'])


library(dplyr)
# Selecting data with city SJ and dropping city and start of week columns
train_sj <- train_total %>% filter(city == 'sj')
train_sj_dropped <- train_sj %>% select(-c(1,4))

# Selecting data with city IQ and dropping city and start of week columns  
train_iq <- train_total %>% filter(city == 'iq')
train_iq_dropped <- train_iq %>% select(-c(1,4))

# Selecting TEST data with city IQ and dropping city and start of week columns  
test_iq <- test_data %>% filter(city == 'iq')
test_iq_original = test_iq

# Selecting TEST data with city SJ and dropping city and start of week columns  
test_sj <- test_data %>% filter(city == 'sj')
test_sj_original = test_sj

# Replacing NA with mean value of the columns in train_iq_dropped
for(i in 1:ncol(train_iq_dropped)){
  train_iq_dropped[,i]=ifelse(is.na(train_iq_dropped[,i]),
                  ave(train_iq_dropped[,i],FUN=function(y) mean(y, na.rm = TRUE)),
                  train_iq_dropped[,i])
}

# Replacing NA with mean value of the columns in train_sj_dropped
for(i in 1:ncol(train_sj_dropped)){
  train_sj_dropped[,i]=ifelse(is.na(train_sj_dropped[,i]),
                              ave(train_sj_dropped[,i],FUN=function(y) mean(y, na.rm = TRUE)),
                              train_sj_dropped[,i])
}

# Replacing NA with mean value of the columns in test_iq_dropped
for(i in 1:ncol(test_iq)){
  test_iq[,i]=ifelse(is.na(test_iq[,i]),
                              ave(test_iq[,i],FUN=function(y) mean(y, na.rm = TRUE)),
                              test_iq[,i])
}

# Replacing NA with mean value of the columns in test_sj_dropped
for(i in 1:ncol(test_sj)){
  test_sj[,i]=ifelse(is.na(test_sj[,i]),
                             ave(test_sj[,i],FUN=function(y) mean(y, na.rm = TRUE)),
                             test_sj[,i])
}

# Correlation of Total Cases with all the columns
cor(train_sj_dropped['total_cases'], train_sj_dropped)
cor(train_iq_dropped['total_cases'], train_iq_dropped)


# Feature Selection
library(leaps)
fit = regsubsets(total_cases ~ .,data = train_sj_dropped, nvmax = 25)
summary(fit)
summary(fit)$rsq # With more features, R squared is also increasing, which is not very helpful
plot(fit, scale = 'bic', main = 'SJ Dataset') # BIC plot penalizes us for more number of features
# Here, 3-4 features will give best 

fit_iq = regsubsets(total_cases ~. , data = train_iq_dropped, nvmax = 25) 
summary(fit_iq)
summary(fit_iq)$rsq
plot(fit_iq, scale = 'bic', main = 'IQ Dataset')


# Linear model for sj dataset

lm_sj = lm(total_cases ~ weekofyear + ndvi_nw + reanalysis_max_air_temp_k + station_diur_temp_rng_c, data = train_sj_dropped)
summary(lm_sj)
plot(hatvalues(lm_sj), main =  'Leverage for Lm_SJ model')
plot(cooks.distance(lm_sj), main =  "Cook's  Distance for Lm_SJ model")
plot(qqnorm(residuals(lm_sj)), main= 'Normality for Residuals of LM_SJ')
acf(residuals(lm_sj))

# Linear model for iq dataset

lm_iq = lm(total_cases ~ year + reanalysis_tdtr_k , data = train_iq_dropped)
summary(lm_iq)
plot(hatvalues(lm_iq), main= 'Leverage for Lm_IQ model')
plot(cooks.distance(lm_iq), main = "Cook's Distance for LM_IQ model")
plot(qqnorm(residuals(lm_iq)), main= 'Normality for Residuals of LM_IQ')
acf(residuals(lm_iq))

test_sj_original['total_cases'] = predict(lm_sj, test_sj)
test_iq_original['total_cases'] = predict(lm_iq, test_iq)


test_combined = rbind(test_sj_original, test_iq_original)
submission = test_combined %>% select(c(1,2,3,25))
submission['total_cases'] = (round(submission['total_cases'], 0))
submission['total_cases'] = ifelse(submission$total_cases < 0, 0, submission$total_cases)

write.csv(submission,'submission7.csv')



###################### 
# Time Series Dataset

train_sj_dropped_filtered = train_sj_dropped %>% select(total_cases)
train_sj_ts = ts(train_sj_dropped_filtered, frequency = 52, start = c(1990,18), end = c(2008,17))
plot(train_sj_ts, main = 'TS data for SJ')
plot(decompose(train_sj_ts, type = 'multiplicative'))

train_iq_dropped_filtered = train_iq_dropped %>% select(total_cases)
train_iq_ts = ts(train_iq_dropped_filtered, frequency = 52, start = c(2000, 26), end = c(2010,25))
plot(train_iq_ts)
plot(decompose(train_iq_ts, type = 'multiplicative'))

# Checking whether data is stationary - Dickey Fuller Test
adf.test(train_sj_ts, alternative = 'stationary') 
adf.test(train_iq_ts, alternative = 'stationary') 

# Not Stationary, So use Diff
plot(diff(train_sj_ts), main = 'Plot of SJ dataset with difference')
train_sj_diff = diff(train_sj_ts)
plot(decompose(train_sj_diff, type ='multiplicative'))


plot(diff(train_iq_ts), main = 'Plot of IQ dataset with difference')
train_iq_diff = diff(train_iq_ts)
plot(decompose(train_iq_diff, type = 'multiplicative'))

# Acf, Pacf tests--- 
acf(train_sj_ts) # Geometric
pacf(train_sj_ts) # Significant till lag 9

acf(diff(train_sj_ts)) 
pacf(diff(train_sj_ts))

# Acf, Pacf tests--- AR Model signs
acf(train_iq_ts) # Geometric
pacf(train_iq_ts) # Significant till lag 5

acf(diff(train_iq_ts)) 
pacf(diff(train_iq_ts))

# Model Estimation for SJ -- AIC should be minimum

arima(train_sj_ts, order = c(0,1,0))
arima(train_sj_ts, order = c(1,1,0))
arima(train_sj_ts, order = c(1,1,1))# Minimum, SO best
arima(train_sj_ts, order = c(2,1,1))  
arima(train_sj_ts, order = c(2,1,2))

arima_sj = arima(train_sj_ts, order = c(1,1,1))
tsdiag(arima_sj) # Random, OK.. No autocorrelation from ACF


arima(train_iq_ts, order = c(0,1,0))
arima(train_iq_ts, order = c(0,1,1))
arima(train_iq_ts, order = c(0,1,2)) # Minimun, Best
arima(train_iq_ts, order = c(0,1,3))
arima(train_iq_ts, order = c(1,1,0))

arima_iq = arima(train_iq_ts, order = c(0,1,2))
tsdiag(arima_iq) # Random, OK.. No autocorrelation from ACF

test_sj_original['total_cases'] = round(predict(arima_sj, n.ahead = 260, se.fit = FALSE),0)
test_iq_original['total_cases'] = round(predict(arima_iq, n.ahead = 156, se.fit = FALSE),0)

combined = rbind(test_sj_original[c(1,2,3,25)], test_iq_original[c(1,2,3,25)])

test_combined = rbind(test_sj_original, test_iq_original)
submission = test_combined %>% select(c(1,2,3,25))

write.csv(submission,'submission_arima.csv')
