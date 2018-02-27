#!/usr/bin/Rscript

library(ggplot2)

sycl_filename <- "sycl-ranges-results.csv"

data <- read.csv(sycl_filename, header=TRUE)
data <- subset(data)


slow_data <- subset(data, path == "slow")
fast_data <- subset(data, path == "fast")

data <- subset(fast_data, select = c("device", "benchmark", "size", "version"))
data$speedup <- slow_data$time / fast_data$time

levels(data$device) <- c("CPU", "GPU")

data$size <- data$size / 2
data$size <- factor(data$size)

plot <- ggplot(data = data) +
        geom_bar(aes(y = speedup, x = size, fill = version),
                 stat = "identity",
                 colour = "black",
                 position = position_dodge(),
                 width = 0.8) +
        facet_grid(device~benchmark) +
        theme_bw() +
        theme(legend.position = "none",
              strip.background = element_rect(fill = "white")) +
        scale_fill_brewer("Version", palette = "YlOrRd") +
        xlab("Size (MB)") +
        ylab("Speedup") +
        ggtitle("Benefit from automatic kernel fusion using views") +
        geom_hline(yintercept = 1)

ggsave("speedup_from_fusion.png", plot, height = 3, width = 10)
