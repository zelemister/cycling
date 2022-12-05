set.seed(9271755)
if(!dir.exists("train")){
  dir.create("train")
}
if(!dir.exists("test")){
  dir.create("test")
}
file_names <- list.files("Images", full.names = TRUE)

n_train <- 9e3
train_files <- sample(file_names, n_train, replace = FALSE)
test_files <- setdiff(file_names, train_files)

train_files_new <- gsub("Images", "train", train_files)
test_files_new <- gsub("Images", "test", test_files)

for(i in seq_along(train_files)) {

  file.rename(from = train_files[[i]], to = train_files_new[[i]])

}

for(i in seq_along(test_files)) {

  file.rename(from = test_files[[i]], to = test_files_new[[i]])

}
