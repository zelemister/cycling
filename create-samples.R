set.seed(9271755)

folder = "Images_512/"
if(!dir.exists(paste0(folder, "train"))){
  dir.create(paste0(folder, "train"))
}
if(!dir.exists(paste0(folder, "test"))){
  dir.create(paste0(folder, "test"))
}
file_names <- list.files(folder, full.names = TRUE)
file_names <- file_names[3:length(file_names)]

n_train <- 9e3
train_files <- sample(file_names, n_train, replace = FALSE)
test_files <- setdiff(file_names, train_files)

train_files_new <- gsub(folder, paste0(folder, "train/"), train_files)
test_files_new <- gsub(folder, paste0(folder, "test/"), test_files)

for(i in seq_along(train_files)) {

  file.rename(from = train_files[[i]], to = train_files_new[[i]])

}

for(i in seq_along(test_files)) {

  file.rename(from = test_files[[i]], to = test_files_new[[i]])

}

