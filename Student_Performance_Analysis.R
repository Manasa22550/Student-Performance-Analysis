# =====================================================================
# Student Performance Analysis - Data Analytics Mini Project (R)
# Dataset : StudentsPerformance.csv (Kaggle: Students Performance in Exams)
# Goal    : Find factors linked to exam scores and predict math score
# =====================================================================

# Run this installation command once, then comment it out:
# install.packages(c("ggplot2", "dplyr"))

library(ggplot2)
library(dplyr)

dir.create("plots", showWarnings = FALSE)

# STEP 1: Load and inspect data
df <- read.csv("StudentsPerformance.csv", stringsAsFactors = FALSE)
names(df)
dim(df)
str(df)
head(df)

names(df) <- c("gender", "race", "parent_edu", "lunch", "test_prep",
               "math", "reading", "writing")
summary(df)

# STEP 2: Data cleaning
colSums(is.na(df))
sum(duplicated(df))
df <- df[!duplicated(df), ]

df$gender <- factor(df$gender)
df$race <- factor(df$race)
df$lunch <- factor(df$lunch)
df$test_prep <- factor(df$test_prep)
df$parent_edu <- factor(
  df$parent_edu,
  levels = c("some high school", "high school", "some college",
             "associate's degree", "bachelor's degree", "master's degree")
)

sum(is.na(df$parent_edu))

# Feature engineering
df$avg_score <- round((df$math + df$reading + df$writing) / 3, 1)
df$grade <- cut(
  df$avg_score,
  breaks = c(-Inf, 50, 65, 80, Inf),
  labels = c("Low", "Average", "Good", "Excellent"),
  right = FALSE
)
table(df$grade)

# Outlier check
boxplot.stats(df$math)$out
boxplot.stats(df$reading)$out
boxplot.stats(df$writing)$out

# STEP 3: Exploratory data analysis
summary(df[, c("math", "reading", "writing", "avg_score")])
sapply(df[, c("math", "reading", "writing")], sd)

table(df$gender)
round(prop.table(table(df$gender)) * 100, 1)

aggregate(cbind(math, reading, writing) ~ test_prep,
          data = df, FUN = mean)

df %>%
  group_by(parent_edu) %>%
  summarise(
    students = n(),
    mean_math = round(mean(math), 1),
    mean_avg = round(mean(avg_score), 1)
  )

# STEP 4: Visualisation
p1 <- ggplot(df, aes(x = math)) +
  geom_histogram(binwidth = 5, fill = "steelblue", colour = "white") +
  labs(title = "Distribution of Math Scores",
       x = "Math score", y = "Number of students")
ggsave("plots/01_math_histogram.png", p1, width = 7, height = 5, dpi = 150)

p2 <- ggplot(df, aes(x = test_prep, y = math, fill = test_prep)) +
  geom_boxplot(show.legend = FALSE) +
  labs(title = "Math Score by Test Preparation Course",
       x = "Test preparation", y = "Math score")
ggsave("plots/02_math_by_test_prep.png", p2, width = 7, height = 5, dpi = 150)

p3 <- ggplot(df, aes(x = parent_edu)) +
  geom_bar(fill = "darkorange") +
  coord_flip() +
  labs(title = "Number of Students by Parental Education",
       x = "Parental education", y = "Count")
ggsave("plots/03_parent_edu_counts.png", p3, width = 7, height = 5, dpi = 150)

p4 <- ggplot(df, aes(x = reading, y = writing)) +
  geom_point(alpha = 0.5, colour = "steelblue") +
  geom_smooth(method = "lm", se = FALSE, colour = "red") +
  labs(title = "Reading vs Writing Score",
       x = "Reading score", y = "Writing score")
ggsave("plots/04_reading_vs_writing.png", p4, width = 7, height = 5, dpi = 150)

p5 <- ggplot(df, aes(x = parent_edu, y = avg_score)) +
  geom_boxplot(fill = "lightgreen") +
  coord_flip() +
  labs(title = "Average Score by Parental Education",
       x = "Parental education", y = "Average score")
ggsave("plots/05_avg_by_parent_edu.png", p5, width = 7, height = 5, dpi = 150)

p6 <- ggplot(df, aes(x = lunch, fill = grade)) +
  geom_bar(position = "fill") +
  labs(title = "Grade Distribution by Lunch Type",
       x = "Lunch type", y = "Proportion", fill = "Grade")
ggsave("plots/06_grade_by_lunch.png", p6, width = 7, height = 5, dpi = 150)

# STEP 5: Hypothesis tests (alpha = 0.05)
qqnorm(df$math)
qqline(df$math)
shapiro.test(df$math)
t.test(math ~ test_prep, data = df)

tab <- table(df$lunch, df$grade)
tab
chisq.test(tab)

round(cor(df[, c("math", "reading", "writing")]), 2)
cor.test(df$reading, df$writing)

anova_model <- aov(math ~ parent_edu, data = df)
summary(anova_model)
TukeyHSD(anova_model)

# STEP 6: Regression - predict math score
set.seed(123)
n <- nrow(df)
train_idx <- sample(seq_len(n), size = round(0.8 * n))
train <- df[train_idx, ]
test <- df[-train_idx, ]

model <- lm(math ~ reading + writing + gender + lunch + test_prep + parent_edu,
            data = train)
summary(model)

pred <- predict(model, newdata = test)
rmse <- sqrt(mean((test$math - pred)^2))
mae <- mean(abs(test$math - pred))
r2 <- 1 - sum((test$math - pred)^2) /
  sum((test$math - mean(test$math))^2)

cat("RMSE:", round(rmse, 2), "\nMAE :", round(mae, 2),
    "\nR2  :", round(r2, 3), "\n")

png("plots/07_regression_diagnostics.png", width = 900, height = 900)
par(mfrow = c(2, 2))
plot(model)
par(mfrow = c(1, 1))
dev.off()

p8 <- ggplot(data.frame(actual = test$math, predicted = pred),
             aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.6, colour = "steelblue") +
  geom_abline(slope = 1, intercept = 0, colour = "red",
              linetype = "dashed") +
  labs(title = "Actual vs Predicted Math Score (Test Set)",
       x = "Actual math score", y = "Predicted math score")
ggsave("plots/08_actual_vs_predicted.png", p8, width = 7, height = 5, dpi = 150)

# STEP 7: Conclusions
# Fill these in after running the script using your actual output:
# 1. Which group scored higher on math: test prep completed or not?
# 2. Is lunch type related to grade category?
# 3. How strong is the reading-writing correlation?
# 4. Does parental education affect math score?
# 5. How good is the regression model? Report Adjusted R2, RMSE and MAE.
# 6. Limitations: one dataset, correlation is not causation, only 3 subjects.
