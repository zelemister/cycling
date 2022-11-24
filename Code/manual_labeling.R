library(png)

#Legend: 0, no bikelane, 1 bikelane, 2 RIM, 3 unsure

#set the paths correctly
project_folder = "C:/Users/Daniel/PycharmProjects/cycling/"
image_folder = paste0(project_folder, "/Data")
list = list.files(image_folder)

#Daniel has to set to true
Daniel = FALSE


set.seed(12345)
random_list = sample(list, length(list))
#Daniel works back to front with the reversed list
if (Daniel){
  random_list = rev(random_list)
}


#labellists
intern_labels = paste0(project_folder, "/labeling_clean.csv")
daniel_labels = paste0(project_folder, "/labeling_daniel.csv")
steffen_labels = paste0(project_folder, "/labeling_steffen.csv")

if (!file.exists(daniel_labels)) {
  file.create(daniel_labels)
  daniel_labels_data = data.frame(Name = random_list, Label = NA)
}else daniel_labels_data = read.csv(daniel_labels)

if (!file.exists(steffen_labels)){
  file.create(steffen_labels)
  steffen_labels_data = data.frame(Name = random_list, Label = NA)
}else steffen_labels_data = read.csv(steffen_labels)

intern_labels_data = read.csv(intern_labels)


for (image in random_list){
  image_has_no_label = ifelse(is.na(steffen_labels_data[steffen_labels_data$Name==image,]$Label) & is.na(daniel_labels_data[daniel_labels_data$Name==image,]$Label), TRUE, FALSE)
  image_has_no_label = image_has_no_label | ifelse(gsub("\\.png$", "", image) %in% intern_labels_data$Name, TRUE, FALSE)
  if (image_has_no_label){
    pp <- readPNG(paste0(image_folder, "/", image))
    plot.new() 
    rasterImage(pp,0,0,1,1)
    input = readline(prompt = "0,1,2,3 or quit ")
    if (input %in% 0:3){
      class = input
      if (Daniel) daniel_labels_data[daniel_labels_data$Name == image,]$Label = class
      else steffen_labels_data[steffen_labels_data$Name == image,]$Label = class
    }else if (input=="quit") break
  }
}

write.csv(daniel_labels_data, daniel_labels, row.names = FALSE)
write.csv(steffen_labels_data, steffen_labels, row.names = FALSE)
