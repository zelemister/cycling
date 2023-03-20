library(dplyr)
library(png)

#manually labeling 9% of the unlabeled predictions

folder = "../Results/"
experiment = "Complete_RIM_Oneshot_Tuned61_Weight100/"
path = paste0(folder, experiment)
data = read.csv(paste0(path, "/unlabeled_predictions_1.csv"))
if(!file.exists(file=paste0(path, "/manual_labels_1.csv"))){
  filtered_data = data[1:round(0.09*nrow(data)),]
  filtered_data$manual_label = rep(NA, nrow(filtered_data))
}else{
  filtered_data = read.csv(file=paste0(path, "/manual_labels_1.csv"))
}
for(i in 1:nrow(filtered_data)){
  if(is.na(filtered_data$manual_label[i])){
    file_path = paste0(folder, "../Images_512/", filtered_data$Name[i])
    pp <- readPNG(file_path)
    plot.new() 
    rasterImage(pp,0,0,1,1)
    print(filtered_data$Name[i])
    input = readline(prompt = "0,1 or quit ")
    if(input == "quit") break
    filtered_data$manual_label[i] <- as.integer(input)
  }
}
write.csv(filtered_data, file=paste0(path, "/manual_labels_1.csv"))