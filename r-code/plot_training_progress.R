library("dplyr")
library("ggplot2")

# this file plots the training progress of one fold of one experiment
path = "../Results/Bikelane_tunedFor2Phase\\Config_1"
file = "/fold_5.csv"

data = read.csv(paste0(path, file))
train_loss = data$train_loss[1]
val_loss = data$val_loss[1]

for(i in 2:nrow(data)){
  train_loss[i] <- min(train_loss[i-1], data$train_loss[i])
  val_loss[i] <- min(val_loss[i-1], data$val_loss[i])
  
}
id = 1:nrow(data)
losses = data.frame(id = id, train_loss = train_loss, val_loss = val_loss)

plot = ggplot(losses, aes(x=id)) +
          geom_step(aes(y=train_loss, color = "Training Loss"))+
          geom_step(aes(y=val_loss, color = "Validation Loss"))+
          ylab("Loss") + 
          xlab("Number of Epochs")
plot
ggsave("Bikelane_tuned_training_progress.png", plot=plot, device = "png", 
       path="C:\\Users\\Daniel\\Desktop\\Consulting", width =5)
