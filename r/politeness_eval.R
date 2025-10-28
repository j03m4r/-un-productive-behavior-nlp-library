#!/usr/bin/env Rscript
suppressWarnings(suppressMessages({
  library(jsonlite)
  library(spacyr)
  library(politeness)
}))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) stop("No text provided")
text <- paste(args, collapse = " ")

# --- Initialize SpaCy ---
spacy_initialize(model = "en_core_web_sm")

# --- Politeness ---
politeness_r <- politeness(
  text,
  parser = "spacy",
  drop_blank = FALSE
)

out <- as.list(politeness_r[1, ])
cat(toJSON(out, auto_unbox = TRUE))