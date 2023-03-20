library(dplyr)

#this file takes the predicted val_predictions_{number}.csv files and creates the quantiles shown in the report

folder = "../cycling/Results/"
experiment = "Complete_RIM_Oneshot_Tuned61_Weight100/Config_1/"
path = paste0(folder, experiment)
files = list.files(path)
files = files[endsWith(files, ".csv")]
files = files[startsWith(files, "val_predictions_")]

#
x = rep(0,3)
d = data.frame("95"=c(), "97.5"=c(),"100"=c())
for(file in files){
  data = read.csv(paste0(path,file))
  
  sorted = data %>% arrange(desc(prediction))
  indizes = which(sorted$label == 1)
  fold_quantiles = quantile(indizes, probs = c(0.95,0.975,1))
  x=x + (fold_quantiles/nrow(sorted))
  d = rbind(d ,sorted$prediction[round(fold_quantiles)])
}
colnames(d) <- c("95", "97.5","100")
d
mean_quantiles = x/5
print(round(mean_quantiles,3))
