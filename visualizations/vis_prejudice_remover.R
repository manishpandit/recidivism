# visualizations for prejudice remover output
library(readr)
library(ggplot2)
library(plyr)

flag = TRUE
base_df <- read_csv('datasets/prejremover-baseline.csv')
pr_df <- read_csv('datasets/prejremover-reg.csv')
cnames <- c('Black Male','Black Female','White Male','White Female','Hispanic Male','Hispanic Female','Other Male','Other Female')

if(flag==TRUE) { # weird bugs with 0 as class. wrap to make sure we only change it once
  base_df$class = base_df$class+1
  pr_df$class = base_df$class+1
  flag = FALSE
}

#base_cnums <- table(base_df$class)
#base_cw <- 1 / base_cnums # each class weight sums to 1
#base_df$weight <- base_cw[base_df$class]
base_df$class_name <- cnames[base_df$class]
base_mu <- ddply(base_df, "class_name", summarize, grp.mean=mean(score))

ggplot(base_df, aes(x=score, y=stat(width*density), fill=score, group=class_name)) + theme_bw() +
  stat_bin(bins=20) + facet_wrap(~class_name,nrow=2) + 
  scale_y_continuous(labels=scales::percent_format(accuracy=2)) +
  geom_vline(data=base_mu, aes(xintercept=grp.mean), linetype="dashed") +
  theme(plot.title = element_text(hjust = 0.5)) + ggtitle("Class Scores, Baseline")
ggsave('visualizations/baseline_class_scores.pdf')
ggsave('visualizations/baseline_class_scores.png')

pr_df$class_name <- cnames[base_df$class]
pr_mu <- ddply(pr_df, "class_name", summarize, grp.mean=mean(score))

ggplot(pr_df, aes(x=score, y=stat(width*density), group=class_name)) + theme_bw() +
  stat_bin(bins=20) + facet_wrap(~class_name, nrow=2) + 
  scale_y_continuous(labels=scales::percent_format(accuracy=2)) +
  geom_vline(data=pr_mu, aes(xintercept=grp.mean), linetype="dashed") +ylab('percent') +
  theme(plot.title = element_text(hjust = 0.5)) + ggtitle("Class Scores, Prejudice Remover")
ggsave('visualizations/pr_class_scores.pdf')
ggsave('visualizations/pr_class_scores.png')

# incorporate adversary data
ad_df <- read_csv('visualizations/NNwithAdversary-woadv.csv')
ad_df$score <- 1 / (1+exp(-ad_df$score))
ad_df$class = 0
ad_df[ad_df$sex == 'Male',]$class <- 1
ad_df[ad_df$sex == 'Female',]$class <- 2
ad_df[ad_df$race == 'White',]$class = ad_df[ad_df$race == 'White',]$class + 2
ad_df[ad_df$race == 'Hispanic',]$class = ad_df[ad_df$race == 'Hispanic',]$class + 4
ad_df[ad_df$race == 'Other',]$class = ad_df[ad_df$race == 'Other',]$class + 6
ad_df$class_name <- cnames[ad_df$class]
ad_df <- data.frame(id=ad_df$id, class=ad_df$class, score=ad_df$score, label=ad_df$true_label, class_name=ad_df$class_name)
ad_mu <- ddply(ad_df, "class_name", summarize, grp.mean=mean(score))
# plot combined results

base_df$model = 'Baseline'
pr_df$model = 'Regularization'
ad_df$model = 'Adversary'
base_mu$model = 'Baseline'
pr_mu$model = 'Regularization'
ad_mu$model = 'Adversary'
all_df <- rbind(base_df, pr_df, ad_df)
all_mu <- rbind(base_mu, pr_mu, ad_mu)
ggplot(all_df, aes(x=score, y=stat(width*density), fill=model, group=model)) + theme_bw() +
  stat_bin(bins=10, position="dodge") + facet_wrap(~class_name, nrow=4) + 
  scale_y_continuous(labels=scales::percent_format(accuracy=2)) +
  scale_x_continuous(labels=c(0,0.2,0.4,0.6,0.8,1), breaks=c(0,0.2,0.4,0.6,0.8,1)) +
  geom_vline(data=all_mu, aes(xintercept=grp.mean, col=model), linetype="dashed") + 
  ylab('Percent') + xlab('Prediction')
ggsave('visualizations/all_class_scores.pdf', width=12, height=10)
ggsave('visualizations/all_class_scores2.png', width=11.77, height=9.9)

# part 2, show training run results
library(reshape2)
avg_df <- read_csv('datasets/prejremover-average-metrics.csv')
bm_df <- read_csv('datasets/prejremover-blackmale-metrics.csv')

avg_df <- melt(avg_df, id.vars=c('eta'), variable.name = "Metric", value.name = "Score")
avgf_df <- avg_df[avg_df$eta < 250,] # filter out eta<250
avgf_df[avgf_df$Metric  != 'Balanced Accuracy',]$Score = -avgf_df[avgf_df$Metric != 'Balanced Accuracy',]$Score


ggplot(avgf_df, aes(x=eta, y=Score, col=Metric, group=Metric)) + geom_line(linetype="dashed") + theme_bw() +
  scale_y_continuous(labels=scales::percent_format(accuracy=2)) +
  ylab("Metric Value, Weighted Avg (over classes)") + xlab("eta (Prejudice Remover)")
ggsave('visualizations/pr_wavg_metrics.pdf')
ggsave('visualizations/pr_wavg_metrics.png')
  
bm_df <- melt(bm_df, id.vars=c('eta'), variable.name = 'Metric', value.name = "Score")
bmf_df <- bm_df[bm_df$eta < 250,]

ggplot(bm_df, aes(x=eta, y=Score, col=Metric, group=Metric)) + geom_line(linetype="dashed") + theme_bw() +
  scale_y_continuous(labels=scales::percent_format(accuracy=2)) +
  ylab("Metric Value, Black Males") + xlab("eta (Prejudice Remover)")
ggsave('visualizations/pr_bm_metrics.pdf')
ggsave('visualizations/pr_bm_metrics.png')
