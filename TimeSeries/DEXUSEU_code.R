#CODE FOR IE594 FINAL PROJECT
#TEAM: Francesco Donato & Niko Stepniak
#Professor: Lin Li
#Teaching Assistant: Muyue Han
#FALL 2022

#training data
train_pct = 0.90

library(nlme)
library(lmtest)
library(stats)
library(strucchange)

#input data
data = read.table("#PUT YOUR DIRECTORY HERE!!!#/DEXUSEU.csv", sep=",")   #<- define your directory
data_title = "DEXUSEU, monthly"
log_title = "log(DEXUSEU, monthly)"
start_year = 0; start_month = 1; frqy = 12
data_ts = ts(log(data$V2[1:(nrow(data)*train_pct)]), start=c(start_year, start_month), frequency=frqy)
data_ts_full = ts(log(data$V2), start=c(start_year, start_month), frequency=frqy)

#raw data
par(mfrow=c(2,2))
plot(exp(data_ts), ylab="U.S. Dollars to One Euro", main=paste(c(data_title, "Training Data")),
     col="red")
plot(data_ts, ylab="Log U.S. Dollars to One Euro", main=paste(c(log_title, "Training Data")), col="red")
plot(exp(data_ts_full), ylab="U.S. Dollars to One Euro", main=paste(c(data_title, "Full Data Set")), col="blue")
plot(data_ts_full, ylab="Log U.S. Dollars to One Euro", main=paste(c(log_title,"Full Data Set")), col="blue")

#stl decomposition
plot(kitty <-stl(data_ts, s.window=12), main=paste(c("STL Decomposition", log_title)), col="red")
summary(kitty)

#polynomial order for the deterministic trend
trend=time(data_ts)
trendf=time(data_ts_full)
sin_cos = cbind(sin(2*pi*trend/12), cos(2*pi*trend/12),sin(2*pi*trend/6),cos(2*pi*trend/6),
                sin(2*pi*trend/4), cos(2*pi*trend/4),sin(2*pi*trend/3), cos(2*pi*trend/3))
trend2 = trend^2; trend3 = trend^3; trend4 = trend^4; trend5 = trend^5;
trend6 = trend^6; trend7 = trend^7; trend8 = trend^8; trend9 = trend^9;
trend10 = trend^10; trend11 = trend^11; trend12 = trend^12;
trend13 = trend^13; trend14 = trend^14; trend15 = trend^15

trend_model_1 = gls(data_ts ~ trend + sin_cos)
trend_model_2 = gls(data_ts ~ trend + trend2 + sin_cos)
trend_model_3 = gls(data_ts ~ trend + trend2 + trend3 + sin_cos)
trend_model_4 = gls(data_ts ~ trend + trend2 + trend3 + trend4 + sin_cos)
trend_model_5 = gls(data_ts ~ trend + trend2 + trend3 + trend4 + trend5 + sin_cos)
trend_model_6 = gls(data_ts ~ trend + trend2 + trend3 + trend4 + trend5 +
                      trend6 + sin_cos)
trend_model_7 = gls(data_ts ~ trend + trend2 + trend3 + trend4 + trend5 +
                      trend6 + trend7 + sin_cos)
trend_model_8 = gls(data_ts ~ trend + trend2 + trend3 + trend4 + trend5 +
                      trend6 + trend7 + trend8 + sin_cos)
trend_model_9 = gls(data_ts ~ trend + trend2 + trend3 + trend4 + trend5 +
                      trend6 + trend7 + trend8 + trend9 + sin_cos)
trend_model_10 = gls(data_ts ~ trend + trend2 + trend3 + trend4 + trend5 +
                       trend6 + trend7 + trend8 + trend9 + trend10 + sin_cos)
trend_model_11 = gls(data_ts ~ trend + trend2 + trend3 + trend4 + trend5 +
                       trend6 + trend7 + trend8 + trend9 + trend10 + trend11 + sin_cos)
trend_model_12 = gls(data_ts ~ trend + trend2 + trend3 + trend4 + trend5 +
                       trend6 + trend7 + trend8 + trend9 + trend10 + trend11 + trend12 + sin_cos)
trend_model_13 = gls(data_ts ~ trend + trend2 + trend3 + trend4 + trend5 +
                       trend6 + trend7 + trend8 + trend9 + trend10 + trend11 + trend12 + trend13 + sin_cos)
trend_model_14 = gls(data_ts ~ trend + trend2 + trend3 + trend4 + trend5 +
                       trend6 + trend7 + trend8 + trend9 + trend10 + trend11 + trend12 + trend13 +
                       trend14 + sin_cos)

F_2_1 = ((sum(trend_model_1$residuals^2)-sum(trend_model_2$residuals^2))/1) /
  (sum(trend_model_2$residuals^2)/(length(data_ts)-length(trend_model_2$coefficients)))
F_3_2 = ((sum(trend_model_2$residuals^2)-sum(trend_model_3$residuals^2))/1) /
  (sum(trend_model_3$residuals^2)/(length(data_ts)-length(trend_model_3$coefficients)))
F_4_3 = ((sum(trend_model_3$residuals^2)-sum(trend_model_4$residuals^2))/1) /
  (sum(trend_model_4$residuals^2)/(length(data_ts)-length(trend_model_4$coefficients)))
F_5_4 = ((sum(trend_model_4$residuals^2)-sum(trend_model_5$residuals^2))/1) /
  (sum(trend_model_5$residuals^2)/(length(data_ts)-length(trend_model_5$coefficients)))
F_6_5 = ((sum(trend_model_5$residuals^2)-sum(trend_model_6$residuals^2))/1) /
  (sum(trend_model_6$residuals^2)/(length(data_ts)-length(trend_model_6$coefficients)))
F_7_6 = ((sum(trend_model_6$residuals^2)-sum(trend_model_7$residuals^2))/1) /
  (sum(trend_model_7$residuals^2)/(length(data_ts)-length(trend_model_7$coefficients)))
F_8_7 = ((sum(trend_model_7$residuals^2)-sum(trend_model_8$residuals^2))/1) /
  (sum(trend_model_8$residuals^2)/(length(data_ts)-length(trend_model_8$coefficients)))
F_9_8 = ((sum(trend_model_8$residuals^2)-sum(trend_model_9$residuals^2))/1) /
  (sum(trend_model_9$residuals^2)/(length(data_ts)-length(trend_model_9$coefficients)))
F_10_9 = ((sum(trend_model_9$residuals^2)-sum(trend_model_10$residuals^2))/1) /
  (sum(trend_model_10$residuals^2)/(length(data_ts)-length(trend_model_10$coefficients)))
F_11_10 = ((sum(trend_model_10$residuals^2)-sum(trend_model_11$residuals^2))/1) /
  (sum(trend_model_11$residuals^2)/(length(data_ts)-length(trend_model_11$coefficients)))
F_12_11 = ((sum(trend_model_11$residuals^2)-sum(trend_model_12$residuals^2))/1) /
  (sum(trend_model_11$residuals^2)/(length(data_ts)-length(trend_model_12$coefficients)))
F_13_12 = ((sum(trend_model_12$residuals^2)-sum(trend_model_13$residuals^2))/1) /
  (sum(trend_model_13$residuals^2)/(length(data_ts)-length(trend_model_13$coefficients)))
F_14_13 = ((sum(trend_model_13$residuals^2)-sum(trend_model_14$residuals^2))/1) /
  (sum(trend_model_14$residuals^2)/(length(data_ts)-length(trend_model_14$coefficients)))


F_2_1
1-pf(F_2_1,1,length(trend)-length(trend_model_2$coefficients))

F_3_2
1-pf(F_3_2,1,length(trend)-length(trend_model_3$coefficients))

F_4_3
1-pf(F_4_3,1,length(trend)-length(trend_model_4$coefficients))

F_5_4
1-pf(F_5_4,1,length(trend)-length(trend_model_5$coefficients))

F_6_5
1-pf(F_6_5,1,length(trend)-length(trend_model_6$coefficients))

F_7_6
1-pf(F_7_6,1,length(trend)-length(trend_model_7$coefficients))

F_8_7
1-pf(F_8_7,1,length(trend)-length(trend_model_8$coefficients))

F_9_8
1-pf(F_9_8,1,length(trend)-length(trend_model_9$coefficients))

F_10_9
1-pf(F_10_9,1,length(trend)-length(trend_model_10$coefficients))

F_11_10
1-pf(F_11_10,1,length(trend)-length(trend_model_11$coefficients))

F_12_11
1-pf(F_12_11,1,length(trend)-length(trend_model_12$coefficients))

F_13_12
1-pf(F_13_12,1,length(trend)-length(trend_model_13$coefficients))

F_14_13
1-pf(F_14_13,1,length(trend)-length(trend_model_14$coefficients))

#9th ORDER WINS!



#reduce the model size to the smallest appropriate model: start=trend9
trend.m = gls(data_ts ~ trend + trend2 + trend3 + trend4 + trend5 +
                trend6 + trend7 + trend8 + trend9 + sin_cos)


##Parsimonious modeling
# eliminate trend3 
reduce.1 = gls(data_ts ~ trend + trend2 + trend4 + trend5 + trend6 + trend7 + trend8 + trend9 + sin_cos)

F_trend.m_reduce.1 = ((sum(reduce.1$residuals^2)-sum(trend.m$residuals^2))/1) /
  (sum(trend.m$residuals^2)/(length(data_ts)-length(trend.m$coefficients)))
F_trend.m_reduce.1
1-pf(F_trend.m_reduce.1,1,length(trend)-length(trend.m$coefficients))

trend_residuals = reduce.1$residuals
trend_model = reduce.1
par(mfrow=c(1,1))
ts.plot(data_ts, trend_model$fitted, ylab="Log U.S. Dollars to One Euro",
        main=paste(c(log_title, "with Fitting")), col=c("red","blue"),lwd=2)

#adding additional regressors for the model
trend_regressors = cbind(trend, trend2, trend4, trend5, trend6, trend7,
                         sin(2*pi*trend/12), cos(2*pi*trend/12), sin(2*pi*trend/6), cos(2*pi*trend/6),
                         sin(2*pi*trend/4), cos(2*pi*trend/4),
                         sin(2*pi*trend/3), cos(2*pi*trend/3))

RSS_trend = sum(trend_residuals^2)
test_range = c((length(data_ts)+1): length(data_ts_full))

test_xreg = cbind(trendf[test_range], 
                  trendf[test_range]^2, #trendf[test_range]^3,
                  trendf[test_range]^4,trendf[test_range]^5, trendf[test_range]^6,trendf[test_range]^7,
                  sin(2*pi*trendf[test_range]/12), cos(2*pi*trendf[test_range]/12),
                  sin(2*pi*trendf[test_range]/6), cos(2*pi*trendf[test_range]/6),
                  sin(2*pi*trendf[test_range]/4), cos(2*pi*trendf[test_range]/4),
                  sin(2*pi*trendf[test_range]/3), cos(2*pi*trendf[test_range]/3))


#residual plots 
summary(trend_model)
par(mfrow=c(1,1))
plot(trend_residuals, pch=20, col="red", main=paste(data_title,": Trend Adjusted Residuals (Stationary)"))
abline(h=0, col="black", lty="dotted")

#acf and pacf for residuals
par(mfrow=c(1,2))
acf(trend_residuals,(length(data_ts))/20, col="red") #use 5% of data for autocorrelations
pacf(trend_residuals,(length(data_ts))/20, col="red")#use 5% of data for autocorrelations

#Modeling Procedure
#AR(1)
arima_1_0_0 = arima(trend_residuals, order=c(1,0,0))
arima_1_0_0
RSS_1_0_0 = arima_1_0_0$sigma2*length(arima_1_0_0$residuals)
RSS_1_0_0

#ARIMA(2,0,1)
n=1; p=2*n; d=0; q=2*n-1
arima_2_0_1 = arima(trend_residuals, order=c(p,d,q))
arima_2_0_1
RSS_2_0_1 = arima_2_0_1$sigma2*length(arima_2_0_1$residuals)
RSS_2_0_1
F_stat = ((RSS_1_0_0 - RSS_2_0_1)/2)/(RSS_2_0_1/(length(trend_residuals)-4*n))
1-pf(F_stat,4,length(trend_residuals)-4*n)

#ARIMA(4,0,3)
n=2; p=2*n; d=0; q=2*n-1
arima_4_0_3 = arima(trend_residuals, order=c(p,d,q), method="CSS")
arima_4_0_3
RSS_4_0_3 = arima_4_0_3$sigma2*length(arima_4_0_3$residuals)
RSS_4_0_3
F_stat = ((RSS_2_0_1 - RSS_4_0_3)/4)/(RSS_4_0_3/(length(trend_residuals)-4*n))
1-pf(F_stat,4,length(trend_residuals)-4*n)


#ARMA specification
n=1; p=2*n; d=0; q=2*n-1
arima_2_0_1 = arima(trend_residuals, order=c(p,d,q))
arima_2_0_1                                                     
RSS_2_0_1 = arima_2_0_1$sigma2*length(arima_2_0_1$residuals)
F_stat = ((RSS_1_0_0 - RSS_2_0_1)/2)/(RSS_2_0_1/(length(trend_residuals)-4*n))
1-pf(F_stat,4,length(trend_residuals)-4*n)

n=1; p=2*n; d=0; q=0 
arima_2_0_0 = arima(trend_residuals, order=c(p,d,q))
arima_2_0_0
RSS_2_0_0 = arima_2_0_0$sigma2*length(arima_2_0_0$residuals)
RSS_2_0_0
F_stat = ((RSS_2_0_0 - RSS_2_0_1)/1)/(RSS_2_0_1/(length(trend_residuals)-4*n))
1-pf(F_stat,1,length(trend_residuals)-4*n)


#Choose ARMA(2,1)

#Integrated model: deterministic trend + ARMA
integrated_model = arima(data_ts, order=c(2,0,1), xreg=trend_regressors)  
#integrated_model = arima(data_ts, order=c(2,0,0), xreg=trend_regressors) #ALSO TRY WITH AR(2)! 
integrated_model

variable_matrix = data.frame(estimate=integrated_model$coef,
                             std_err=rep(0,length(integrated_model$coef)), p_value=rep(0,length(integrated_model$coef)))

for(i in 1:length(integrated_model$coef)){
  variable_matrix[i,2] = sqrt(integrated_model$var.coef[i,i])
  variable_matrix[i,3] = dt(integrated_model$coef[i]/sqrt(integrated_model$var.coef[i,i]),
                            length(data_ts)-length(integrated_model$coef))
}
variable_matrix

#characteristic roots calculation
polyroot(c(1,-1.756503,0.862464))

#integrated_model residual plots
#par(mfrow=c(1,2))
#plot(integrated_model$residuals, col="red", main="Integrated Model Residuals", )
#pacf(integrated_model$residuals, main="PACF: Integrated Model Residuals", col="red")

#forecasting graph
joint_pred = predict(integrated_model, n.ahead=ceiling(length(data_ts_full)*(1-train_pct)), newxreg=test_xreg)
U = joint_pred$pred + 2*joint_pred$se
L = joint_pred$pred - 2*joint_pred$se

par(mfrow=c(1,1))
ts.plot(data_ts_full, joint_pred$pred, col=c("red","dark green"),
        main=paste(c(log_title, "vs Forecast with 95% Confidence Interval")), ylim=c(-1,0.5))
lines(U, col="blue", lty="dashed")
lines(L, col="blue", lty="dashed")
abline(v=length(data_ts)/12, col="black", lty="dotted")

#close-up on forecast period
ts.plot(data_ts_full, joint_pred$pred, col=c("red","dark green"),
        main=paste(c(log_title, "vs Forecast with 95% Confidence Interval")),
        xlim=c(length(data_ts_full)*train_pct/12,length(data_ts_full)/12))
lines(U, col="blue", lty="dashed")
lines(L, col="blue", lty="dashed")
abline(v=length(data_ts)/12, col="black", lty="dotted")


ts.plot(
  data_ts,
  data_ts-integrated_model$residuals,
  col=c("red","blue")
)