
results <- data.frame()

for (dataset in c("empirical", "null", "alternative")) {
  for (model in c("null", "alternative")) {
    file <- here::here("stan", "results", dataset, model, "bridgesampling.csv")
    df <- read.csv(file)
    df[["dataset"]] <- dataset
    df[["model"]] <- model
    results <- rbind(results, df)
  }
}

write.csv(results, file = here::here("stan", "results.csv"), row.names = FALSE)
