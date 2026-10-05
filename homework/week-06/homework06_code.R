# Homework 06: KNN vs. logistic regression (Questions 1 and 2)
# Base R only. Expects data/breast-cancer.csv and data/breast-cancer-split.csv.
# Writes q1_roc.png and q2_roc.png to the working directory.

# ---------- Load data and the provided train/test split ----------
df <- read.csv("data/breast-cancer.csv")
sp <- read.csv("data/breast-cancer-split.csv")
stopifnot(all(sp$row == seq_len(nrow(df))))
is_train <- sp$split == "train"

cols <- c("mean_radius", "mean_texture", "mean_smoothness")
Xtr <- as.matrix(df[is_train,  cols]); ytr <- df$y[is_train]
Xte <- as.matrix(df[!is_train, cols]); yte <- df$y[!is_train]

# Standardize with training means / SDs only
mu <- colMeans(Xtr)
s  <- apply(Xtr, 2, sd)
Xtr <- sweep(sweep(Xtr, 2, mu, "-"), 2, s, "/")
Xte <- sweep(sweep(Xte, 2, mu, "-"), 2, s, "/")

# ---------- Helpers ----------
# KNN estimate of P(y = 1): mean label of the k nearest training points
# (Euclidean distance, equal weights). If `query` is the training matrix,
# each point is its own nearest neighbor (distance 0), as Question 2 requires.
knn_prob <- function(train, ytrain, query, k) {
  d2_train <- rowSums(train^2)
  d2_query <- rowSums(query^2)
  D2 <- outer(d2_query, d2_train, "+") - 2 * query %*% t(train)  # squared distances
  apply(D2, 1, function(d) mean(ytrain[order(d)[1:k]]))
}

# ROC curve and AUC (trapezoidal rule; ties in scores handled by thresholds)
roc_curve <- function(y, p) {
  thr <- sort(unique(p), decreasing = TRUE)
  tpr <- c(0, sapply(thr, function(t) mean(p[y == 1] >= t)))
  fpr <- c(0, sapply(thr, function(t) mean(p[y == 0] >= t)))
  auc <- sum(diff(fpr) * (head(tpr, -1) + tail(tpr, -1)) / 2)
  list(fpr = fpr, tpr = tpr, auc = auc)
}

# ---------- Question 1 ----------
train_df <- data.frame(Xtr, y = ytr)
test_df  <- data.frame(Xte)
fit_lr   <- glm(y ~ ., data = train_df, family = binomial)   # intercept, no penalty

probs <- list(Logistic = predict(fit_lr, newdata = test_df, type = "response"))
for (k in c(5, 15, 25, 35, 45)) {
  probs[[paste0("KNN k=", k)]] <- knn_prob(Xtr, ytr, Xte, k)
}

rocs <- lapply(probs, function(p) roc_curve(yte, p))
cat("Question 1: test AUC\n")
for (n in names(rocs)) cat(sprintf("  %-10s %.4f\n", n, rocs[[n]]$auc))

png("q1_roc.png", width = 1000, height = 1000, res = 150)
cols_q1 <- c("black", "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e")
plot(NA, xlim = c(0, 1), ylim = c(0, 1), xlab = "False positive rate",
     ylab = "True positive rate", main = "Q1: test ROC curves")
abline(0, 1, lty = 2)
for (i in seq_along(rocs)) lines(rocs[[i]]$fpr, rocs[[i]]$tpr, col = cols_q1[i], lwd = 2)
legend("bottomright", bty = "n", col = cols_q1, lwd = 2,
       legend = sprintf("%s (AUC=%.3f)", names(rocs), sapply(rocs, `[[`, "auc")))
dev.off()

# ---------- Question 2 ----------
ks <- c(1, 3, 5, 10, 20)
png("q2_roc.png", width = 2400, height = 500, res = 130)
par(mfrow = c(1, 5), mar = c(4, 4, 3, 1))
cat("Question 2: train / test AUC\n")
for (k in ks) {
  r_tr <- roc_curve(ytr, knn_prob(Xtr, ytr, Xtr, k))   # self included among neighbors
  r_te <- roc_curve(yte, knn_prob(Xtr, ytr, Xte, k))
  cat(sprintf("  k=%-3d train=%.4f  test=%.4f\n", k, r_tr$auc, r_te$auc))
  plot(NA, xlim = c(0, 1), ylim = c(0, 1), xlab = "False positive rate",
       ylab = "True positive rate", main = paste("k =", k))
  abline(0, 1, lty = 2)
  lines(r_tr$fpr, r_tr$tpr, col = "#1b9e77", lwd = 2)
  lines(r_te$fpr, r_te$tpr, col = "#d95f02", lwd = 2)
  legend("bottomright", bty = "n", col = c("#1b9e77", "#d95f02"), lwd = 2,
         legend = c(sprintf("Train AUC=%.3f", r_tr$auc), sprintf("Test AUC=%.3f", r_te$auc)))
}
dev.off()
