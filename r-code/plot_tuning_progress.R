library(jsonlite)
library(ggplot2)
library(dplyr)

#adjust path to smac run whose training progress should be plotted
path = "../Results/smac3"
folder = "bikelane/blaskjdlbaksjdblkasjdbl/0"


#read the json files
json_data = read_json(paste(path,folder,"intensifier.json", sep = "/"))
json_data
run_history = read_json(paste(path,folder,"runhistory.json", sep = "/"))
run_history$stats

id=c()
costs=c()
for(i in 1:length(json_data$trajectory)){
  id[i] = print(json_data$trajectory[[i]]$config_ids[[1]])
  costs[i] = json_data$trajectory[[i]]$costs[[1]]
}

data = data.frame(id=id, costs=costs)

run_history$stats$finished
data[nrow(data)+1,]=c(run_history$stats$finished, data$costs[nrow(data)])

plot = data%>%
                  ggplot(aes(x=id, y=costs)) +
                  geom_step() +
                  xlab("Number of Configurations tried") + 
                  ylab("Configuration Cost")
ggsave("One_shot_tuning_progress.png", plot=plot, device = "png", 
       path="C:\\Users\\Daniel\\Desktop\\Consulting", width =10)
